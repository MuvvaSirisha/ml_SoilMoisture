"""
engine.py
=========
Optimized Soil Moisture Analysis Engine.
Handles operation-specific analysis with flexible output (scalar, map, or both).
"""

import matplotlib
matplotlib.use('Agg')   # Must be set BEFORE importing pyplot

import math
import xarray as xr
import rioxarray          # registers the .rio accessor on xarray objects
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
from shapely.geometry import mapping
from datetime import datetime
import warnings
import traceback
warnings.filterwarnings('ignore')

_TREND_CMAP = 'RdBu'
TREND_VMAX = 0.02  # Fixed limit for both SMAP and AMSR trend maps

from Config import (FIGURE_SETTINGS, OPERATIONS,
                    BORDER_SETTINGS, OUTPUT_PATH,
                    MIN_TREND_YEARS, TREND_WARN_YEARS)
from utils import OutputFormatter, DateAnalyzer

# Normalise common spelling variants → exact shapefile STATE values
_STATE_SPELLING_NORM = {
    "chhattisgarh": "chhatisgarh",   # shapefile uses single-t spelling
    "chattisgarh":  "chhatisgarh",
}


# ============================================================================
# BORDER HELPER
# ============================================================================

def _draw_borders(ax, gdf, region_name=None):
    """Overlay actual shapefile borders in black on a map axes."""
    try:
        bs_all    = BORDER_SETTINGS['all_states_border']
        bs_region = BORDER_SETTINGS['region_border']

        gdf.boundary.plot(
            ax=ax, edgecolor='black',
            linewidth=bs_all['linewidth'],
            facecolor='none', zorder=bs_all['zorder']
        )

        if region_name and region_name.lower() != 'india':
            region_gdf = gdf[
                gdf['STATE'].str.strip().str.lower() == region_name.lower()
            ]
            if not region_gdf.empty:
                try:
                    region_gdf.boundary.plot(
                        ax=ax, edgecolor='black',
                        linewidth=bs_region['linewidth'],
                        facecolor='none', zorder=bs_region['zorder']
                    )
                except Exception:
                    pass

    except Exception as e:
        print(f"⚠️  Border drawing note: {e}")


# ============================================================================
# SLOPE MAP HELPERS  (Mann-Kendall + Sen's Slope — yearly-mean based)
# ============================================================================

def _annual_mean(data_array):
    """
    Collapse a DataArray's time dimension into yearly means.
    Returns a DataArray with a 'year' dimension instead of 'time'.
    Handles xarray versions that name the groupby dim differently.
    """
    annual = data_array.groupby('time.year').mean(dim='time')
    # Normalise the groupby result dimension to always be called 'year'
    for possible in ['time.year', 'year']:
        if possible in annual.dims:
            if possible != 'year':
                annual = annual.rename({possible: 'year'})
            break
    return annual


def _sens_slope(vals):
    """
    Theil-Sen estimator: median of all pairwise slopes.
    This is the standard robust slope estimator paired with Mann-Kendall.
    Returns slope in the same units per time-step (m³/m³ / year).
    """
    n = len(vals)
    if n < 2:
        return np.nan
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            slopes.append((vals[j] - vals[i]) / (j - i))
    return float(np.median(slopes))


def _mann_kendall_pval_tau(vals):
    """
    Mann-Kendall trend test using scipy.stats.kendalltau.
    Returns (p_value, tau) — non-parametric, robust to outliers.
    Standard method in WMO / hydrology for monotonic trend detection.
    """
    n = len(vals)
    if n < 2:
        return 1.0, 0.0
    tau, p = stats.kendalltau(np.arange(n), vals)
    return float(p), float(tau)


def _compute_spatial_slope_and_pval(annual_da):
    """
    Pixel-wise Mann-Kendall + Sen's Slope on an annual-mean DataArray.

    Parameters
    ----------
    annual_da : xr.DataArray  with dimension 'year'

    Returns
    -------
    slope_da  : xr.DataArray  — Sen's slope in m³/m³ per year
    pval_da   : xr.DataArray  — Mann-Kendall p-value
    """
    # Make sure the year dim exists
    if 'year' not in annual_da.dims:
        raise ValueError(
            f"_compute_spatial_slope_and_pval: expected 'year' dim, "
            f"got {list(annual_da.dims)}"
        )

    def _px_sens(y):
        """Per-pixel Sen's slope (Theil-Sen estimator)."""
        m = ~np.isnan(y)
        n_valid = int(m.sum())
        if n_valid >= MIN_TREND_YEARS:
            return _sens_slope(y[m])
        return np.nan

    def _px_mk_pval(y):
        """Per-pixel Mann-Kendall p-value."""
        m = ~np.isnan(y)
        n_valid = int(m.sum())
        if n_valid >= MIN_TREND_YEARS:
            p, _ = _mann_kendall_pval_tau(y[m])
            return p
        return np.nan

    slope_da = xr.apply_ufunc(
        _px_sens, annual_da,
        input_core_dims=[['year']], vectorize=True
    )
    pval_da = xr.apply_ufunc(
        _px_mk_pval, annual_da,
        input_core_dims=[['year']], vectorize=True
    )
    return slope_da, pval_da


def _add_stippling(ax, slope_da, pval_da, threshold=0.05, dot_size=5):
    """
    Add stippling dots on statistically significant pixels.
    Safe: silently skips if coordinates or shapes are inconsistent.

    Parameters
    ----------
    ax        : matplotlib Axes
    slope_da  : xr.DataArray  (y, x)  — slope map
    pval_da   : xr.DataArray  (y, x)  — p-value map
    threshold : float  — significance level (default 0.05)
    dot_size  : float  — scatter marker size
    """
    try:
        sig_mask = (pval_da < threshold)
        if not bool(sig_mask.any()):
            return
            
        lon_dim = 'lon' if 'lon' in pval_da.coords else 'x'
        lat_dim = 'lat' if 'lat' in pval_da.coords else 'y'

        # Safely convert to a flat dataframe to perfectly match coordinates
        df = sig_mask.to_dataframe(name='sig').reset_index()
        df_sig = df[df['sig'] == True]
        
        if df_sig.empty:
            return
            
        ax.scatter(
            df_sig[lon_dim], df_sig[lat_dim],
            s=dot_size, c='black', marker='.', linewidths=0,
            label=f'p < {threshold} (significant)', zorder=5
        )
        ax.legend(fontsize=9, loc='lower right',
                  markerscale=2, framealpha=0.7)
    except Exception as e:
        print(f"⚠️  Stippling skipped: {e}")


def _compute_spatial_slope(data_array):
    """
    Legacy wrapper (daily-index slope) kept for any callers that still need it.
    New trend code should call _compute_spatial_slope_and_pval on annual data.
    """
    def get_px_slope(y):
        idx = np.arange(len(y))
        m   = ~np.isnan(y)
        return stats.linregress(idx[m], y[m])[0] if sum(m) > 2 else np.nan

    return xr.apply_ufunc(
        get_px_slope, data_array,
        input_core_dims=[['time']], vectorize=True
    )


# ============================================================================
# ENGINE
# ============================================================================

class SM_Engine:
    """
    Optimized Soil Moisture Analysis Engine.

    v2.6 fixes:
      - All comparison types (2-way region, N-way time) use _visualize_n_panel
        so they ALWAYS produce exactly ONE combined output image.
      - Map generation is more robust with full error reporting.

    v2.8 fixes:
      - Multi-period slope/trend queries produce ONE combined image.
      - New execute_analysis_batch() for main.py to collect N ranges together.
    """

    def __init__(self):
        from cloud.dataset_manager import get_full_dataset
        from cloud.shapefile_manager import get_shapefile_path

        print("📂 Syncing and Loading Zarr datasets from Google Drive...")
        self.ds = get_full_dataset()

        for d in list(self.ds.dims):
            if 'lat' in d.lower():
                self.ds = self.ds.rename({d: 'y'})
            if 'lon' in d.lower():
                self.ds = self.ds.rename({d: 'x'})

        self.ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        if self.ds.rio.crs is None:
            self.ds.rio.write_crs("EPSG:4326", inplace=True)

        print("📍 Syncing and Loading shapefile from Google Drive...")
        self.shp_path = get_shapefile_path()
        self.gdf = gpd.read_file(self.shp_path)
        if self.gdf.crs is None or self.gdf.crs.to_epsg() != 4326:
            self.gdf = self.gdf.to_crs("EPSG:4326")

        self.available_regions = (
            self.gdf['STATE'].str.strip().str.lower().unique()
        )
        print(f"✅ Engine ready! Available regions: {len(self.available_regions)}")

    # ------------------------------------------------------------------ #
    # PUBLIC: execute_analysis                                             #
    # ------------------------------------------------------------------ #

    def execute_analysis(self, region, start_date, end_date, operation,
                         output_type='both', comparison_info=None):
        """
        Execute the requested analysis for a SINGLE date range.
        Returns (result_message: str, visualization_created: bool)
        """
        if start_date and len(start_date) == 7:
            start_date += "-01"
        if end_date and len(end_date) == 7:
            end_date += "-28"

        print(f"📊 Processing [{operation}] for [{region}]  "
              f"{start_date} → {end_date} | output_type={output_type} ...")

        if operation == 'comparison':
            return self._handle_comparison(
                region, start_date, end_date, output_type, comparison_info
            )

        subset = self.ds.sel(time=slice(start_date, end_date)).compute()
        clipped, display_region, ok = self._clip_region(subset, region)
        if not ok:
            return clipped, False

        v = list(self.ds.data_vars)[0]

        if clipped.time.size == 0 or int(clipped[v].count()) == 0:
            return f"❌ No data available for {display_region} ({start_date} to {end_date})", False

        if operation == 'mean':
            return self._analyze_mean(
                clipped, v, display_region, region,
                start_date, end_date, output_type)
        elif operation == 'slope':
            return self._analyze_slope(
                clipped, v, display_region, region,
                start_date, end_date, output_type)
        elif operation == 'minimum':
            return self._analyze_minimum(
                clipped, v, display_region, region,
                start_date, end_date, output_type)
        elif operation == 'maximum':
            return self._analyze_maximum(
                clipped, v, display_region, region,
                start_date, end_date, output_type)
        else:
            return f"❌ Unknown operation: {operation}", False

    # ------------------------------------------------------------------ #
    # PUBLIC: execute_analysis_batch  (v2.8 — NEW)                        #
    # ------------------------------------------------------------------ #

    def execute_analysis_batch(self, region, date_ranges, operation,
                               output_type='both'):
        """
        Execute the same non-comparison operation over N date ranges and
        produce ONE combined output image (for slope/trend multi-period).

        Parameters
        ----------
        region      : str
        date_ranges : list of (start_date, end_date) tuples
        operation   : 'slope' | 'mean' | 'minimum' | 'maximum'
        output_type : 'scalar' | 'map' | 'both'

        Returns
        -------
        list of (result_message: str, viz_created: bool) — one per range.
        The visualization for ALL ranges is saved once to OUTPUT_PATH.
        """
        if not date_ranges:
            return []

        v = list(self.ds.data_vars)[0]

        # ── Collect per-period data + scalar results ───────────────────
        results        = []   # (msg, False) per period — viz handled jointly
        clipped_list   = []   # clipped DataArrays per period
        label_list     = []   # human-readable label per period
        display_region = region.title()

        for s, e in date_ranges:
            if s and len(s) == 7: s += "-01"
            if e and len(e) == 7: e += "-28"

            subset = self.ds.sel(time=slice(s, e)).compute()
            clipped, disp, ok = self._clip_region(subset, region)
            if not ok:
                results.append((clipped, False))
                clipped_list.append(None)
                label_list.append(f"{s} to {e}")
                continue

            display_region = disp
            clipped_list.append(clipped)
            label_list.append(f"{s}\nto {e}")

            # Scalar result for this period (no map yet)
            msg, _ = self._analyze_single_no_viz(
                clipped, v, disp, region, s, e, operation, output_type
            )
            results.append((msg, False))  # viz_created=False for now

        # ── Render ONE combined image for all periods ──────────────────
        viz_created = False
        if output_type in ['map', 'both']:
            valid_pairs = [
                (c, l) for c, l in zip(clipped_list, label_list) if c is not None
            ]
            if valid_pairs:
                valid_clipped, valid_labels = zip(*valid_pairs)
                if operation == 'slope':
                    viz_created = self._visualize_slope_n_panel(
                        clipped_list   = list(valid_clipped),
                        var_name       = v,
                        labels         = list(valid_labels),
                        region_name    = region,
                        display_region = display_region,
                    )
                else:
                    # For mean / min / max — use existing _visualize_n_panel
                    maps = [self._metric_map(c[v], operation) for c in valid_clipped]
                    viz_created = self._visualize_n_panel(
                        maps         = maps,
                        labels       = list(valid_labels),
                        region_name  = region,
                        region_names = None,
                        suptitle     = (
                            f"{len(valid_labels)}-Period {operation.upper()}: "
                            f"{display_region}"
                        ),
                        metric       = operation,
                    )

        # Mark the last result as viz_created so main.py prints one notice
        if results:
            last_msg, _ = results[-1]
            results[-1] = (last_msg, viz_created)

        return results

    # ── Internal: run analysis WITHOUT generating visualization ───────────

    def _analyze_single_no_viz(self, clipped, var_name, display_region,
                                raw_region, start_date, end_date,
                                operation, output_type):
        """
        Run a single-period analysis and return scalar output only.
        Visualization is suppressed — the caller handles combined rendering.
        """
        # Force scalar so no individual map is saved
        scalar_output_type = 'scalar' if output_type in ('map', 'both') else output_type

        if operation == 'slope':
            return self._analyze_slope(
                clipped, var_name, display_region, raw_region,
                start_date, end_date, scalar_output_type
            )
        elif operation == 'mean':
            return self._analyze_mean(
                clipped, var_name, display_region, raw_region,
                start_date, end_date, scalar_output_type
            )
        elif operation == 'minimum':
            return self._analyze_minimum(
                clipped, var_name, display_region, raw_region,
                start_date, end_date, scalar_output_type
            )
        elif operation == 'maximum':
            return self._analyze_maximum(
                clipped, var_name, display_region, raw_region,
                start_date, end_date, scalar_output_type
            )
        return "❌ Unknown operation.", False

    # ------------------------------------------------------------------ #
    # REGION CLIPPING                                                      #
    # ------------------------------------------------------------------ #

    def _clip_region(self, subset, region):
        """Clip dataset to a region. Returns (clipped, display_name, success)."""
        # Normalise common spelling variants to match shapefile STATE values
        norm_region = _STATE_SPELLING_NORM.get(region.lower(), region.lower())

        if norm_region == 'india':
            clipped = subset.rio.clip(
                self.gdf.geometry.apply(mapping),
                self.gdf.crs, drop=True
            )
            return clipped, "India (All States)", True

        ZONES = {
            'north': ['chandigarh', 'delhi', 'haryana', 'himachal pradesh', 'jammu and kashmir', 'ladakh', 'punjab', 'rajasthan', 'uttarakhand', 'uttar pradesh'],
            'south': ['andhra pradesh', 'karnataka', 'kerala', 'lakshadweep', 'puducherry', 'tamil nadu', 'telangana', 'andaman & nicobar'],
            'east': ['bihar', 'jharkhand', 'odisha', 'west bengal'],
            'west': ['dadra & nagar haveli & daman & diu', 'goa', 'gujarat', 'maharashtra'],
            'central': ['chhatisgarh', 'madhya pradesh'],
            'northeast': ['arunachal pradesh', 'assam', 'manipur', 'meghalaya', 'mizoram', 'nagaland', 'sikkim', 'tripura']
        }

        if norm_region in ZONES:
            zone_states = ZONES[norm_region]
            region_gdf = self.gdf[
                self.gdf['STATE'].str.strip().str.lower().isin(zone_states)
            ]
            if region_gdf.empty:
                return (f"❌ Zone '{region}' not found.", None, False)
            clipped = subset.rio.clip(
                region_gdf.geometry.apply(mapping),
                self.gdf.crs, drop=True
            )
            return clipped, f"{region.title()} Zone", True

        region_gdf = self.gdf[
            self.gdf['STATE'].str.strip().str.lower() == norm_region
        ]
        if region_gdf.empty:
            available = ", ".join(sorted(self.available_regions)[:8])
            return (f"❌ Region '{region}' not found. "
                    f"Available (first 8): {available}...", None, False)

        clipped = subset.rio.clip(
            region_gdf.geometry.apply(mapping),
            self.gdf.crs, drop=True
        )
        return clipped, region.title(), True

    # ================================================================== #
    # SINGLE OPERATION HANDLERS
    # ================================================================== #

    def _analyze_mean(self, clipped, var_name, display_region, raw_region,
                      start_date, end_date, output_type):
        data  = clipped[var_name]
        total = int(data.size)
        null  = int(data.isnull().sum())
        
        expected_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
        spatial_size = total // max(1, data.time.size) if hasattr(data, 'time') and data.time.size > 0 else 0
        expected_total = expected_days * spatial_size
        
        missing_days_count = max(0, expected_days - (data.time.size if hasattr(data, 'time') else 1))
        true_null = null + (missing_days_count * spatial_size)
        true_total = expected_total if expected_total > 0 else total
        
        stats_dict = {
            'mean':        float(data.mean()),
            'count':       int(data.count()),
            'missing_pct': (true_null / true_total * 100) if true_total else 0
        }
        output = OutputFormatter.format_mean(
            stats_dict, output_type, display_region, start_date, end_date)

        viz_created = False
        if output_type in ['map', 'both']:
            viz_created = self._visualize_mean(
                clipped, var_name, display_region, raw_region,
                start_date, end_date)

        return output, viz_created

    def _visualize_mean(self, clipped, var_name, display_region, raw_region,
                        start_date, end_date):
        try:
            settings  = FIGURE_SETTINGS['mean']
            fig, ax   = plt.subplots(1, 1, figsize=settings['figsize'],
                                     constrained_layout=True)
            mean_data = clipped[var_name].mean(dim='time')
            is_single_day = (start_date == end_date)
            cbar_label = 'Soil Moisture (m³/m³)' if is_single_day else 'Mean Soil Moisture (m³/m³)'
            mean_data.plot(ax=ax, x='x', y='y', cmap='YlGnBu',
                           cbar_kwargs={'label': cbar_label, 'shrink': 0.85})
            _draw_borders(ax, self.gdf, raw_region)

            if is_single_day:
                title_str = f"Soil Moisture: {display_region}\n({start_date})"
            else:
                title_str = f"Mean Soil Moisture: {display_region}\n({start_date} to {end_date})"
            ax.set_title(title_str, fontsize=16, fontweight='bold', pad=10)
            ax.set_xlabel("Longitude", fontsize=13)
            ax.set_ylabel("Latitude",  fontsize=13)
            ax.tick_params(labelsize=11)
            ax.set_aspect('equal')
            plt.savefig(OUTPUT_PATH, dpi=settings['dpi'], bbox_inches='tight', facecolor='white', format='png')
            plt.close()
            print(f"✅ Map saved → {OUTPUT_PATH}")
            return True
        except Exception as e:
            print(f"⚠️  Visualization error (mean): {e}")
            traceback.print_exc()
            plt.close('all')
            return False

    def _analyze_slope(self, clipped, var_name, display_region, raw_region,
                       start_date, end_date, output_type):
        """
        Trend analysis using ANNUAL MEANS — Mann-Kendall + Sen's Slope.

        Steps
        -----
        1. Compute spatial-mean time series (area-averaged).
        2. Group by year → one mean value per year.
        3. Mann-Kendall test for trend direction & significance.
        4. Sen's (Theil-Sen) slope for magnitude in m³/m³/year.
        5. Hard block if n_years < MIN_TREND_YEARS; warning if < TREND_WARN_YEARS.
        """
        # ── Area-averaged annual time series ──────────────────────────
        ts_daily  = clipped[var_name].mean(dim=['x', 'y'])
        annual_ts = ts_daily.groupby('time.year').mean(dim='time')
        years     = annual_ts.year.values
        y_vals    = annual_ts.values
        mask      = ~np.isnan(y_vals)
        n_years   = int(mask.sum())
        x_idx     = np.arange(len(years))

        # ── Hard minimum period guard ─────────────────────────────────
        if n_years < MIN_TREND_YEARS:
            return (
                f"❌ **Trend analysis requires at least {MIN_TREND_YEARS} years of data.**\n\n"
                f"The selected period has only **{n_years} year(s)** of valid annual data "
                f"({start_date[:4]}–{end_date[:4]}).\n\n"
                f"Please extend your date range. "
                f"The WMO recommends ≥ {TREND_WARN_YEARS} years for a reliable trend."
            ), False

        # ── Mann-Kendall test (non-parametric, rank-based) ────────────
        p_val, mk_tau = _mann_kendall_pval_tau(y_vals[mask])

        # ── Sen's slope (Theil-Sen estimator, median-based, robust) ──
        slope = _sens_slope(y_vals[mask])

        # ── OLS intercept for chart baseline (display only) ──────────
        ols_res = stats.linregress(x_idx[mask], y_vals[mask])
        ols_slope = ols_res.slope
        ols_intercept = ols_res.intercept

        n_years_total = len(years)
        missing_n     = n_years_total - n_years
        reliability_warning = n_years < TREND_WARN_YEARS

        stats_dict = {
            'slope':        float(slope),
            'mk_tau':       float(mk_tau),
            'p_value':      float(p_val),
            'total_change': float(slope * n_years),
            'count':        n_years,
            'missing_pct':  (missing_n / n_years_total * 100) if n_years_total else 0,
            'n_years':      n_years,
            'reliability_warning': reliability_warning,
        }

        output = OutputFormatter.format_slope(
            stats_dict, output_type, display_region, start_date, end_date)

        viz_created = False
        if output_type in ['map', 'both']:
            viz_created = self._visualize_slope(
                clipped, var_name,
                annual_ts, years, y_vals,
                slope, ols_slope, ols_intercept, x_idx,
                display_region, raw_region, start_date, end_date,
                mk_tau, p_val, reliability_warning)

        return output, viz_created

    def _visualize_slope(self, clipped, var_name,
                         annual_ts, years, y_vals,
                         sens_slope_val, ols_slope, ols_intercept, x_idx,
                         display_region, raw_region, start_date, end_date,
                         mk_tau, p_val, reliability_warning):
        """
        Single-period slope visualization (2 panels).
        """
        try:
            settings = FIGURE_SETTINGS['slope']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=settings['figsize'],
                                           constrained_layout=True)

            annual_da           = _annual_mean(clipped[var_name])
            spatial_slope, pval = _compute_spatial_slope_and_pval(annual_da)

            spatial_slope.plot(
                ax=ax1, x='x' if 'x' in spatial_slope.coords else 'lon', y='y' if 'y' in spatial_slope.coords else 'lat', cmap='RdBu', center=0,
                cbar_kwargs={"label": "Rate of Change (m\u00b3/m\u00b3/year)", "shrink": 0.85}
            )
            _add_stippling(ax1, spatial_slope, pval, dot_size=3)

            _draw_borders(ax1, self.gdf, raw_region)
            ax1.set_title(
                f"Soil Moisture Trend Map \u2014 {display_region}\n"
                f"Blue = getting wetter  |  Red = getting drier  |  Dots = confirmed trend",
                fontsize=13, fontweight='bold', pad=10
            )
            ax1.set_xlabel("Longitude", fontsize=12)
            ax1.set_ylabel("Latitude",  fontsize=12)
            ax1.tick_params(labelsize=11)
            ax1.set_aspect('equal')

            n_years   = int((~np.isnan(y_vals)).sum())
            direction = "\U0001f4c8 Getting Wetter" if sens_slope_val > 0 else "\U0001f4c9 Getting Drier"

            valid_idx = np.where(~np.isnan(y_vals))[0]
            first_i   = valid_idx[0] if len(valid_idx) else 0
            first_val = y_vals[first_i] if len(valid_idx) else 0.0
            sens_line = first_val + sens_slope_val * (x_idx - first_i)

            ax2.plot(years, y_vals, 'o-',
                     color='#2980b9', linewidth=2.0, markersize=6,
                     alpha=0.85, label='Yearly Moisture')
            ax2.plot(years, sens_line,
                     color='#27ae60', linewidth=2.5, linestyle='-',
                     label='Main Trend Line')
            ax2.plot(years, ols_intercept + ols_slope * x_idx,
                     color='#e74c3c', linewidth=1.5, linestyle='--',
                     alpha=0.6, label='Alternative Estimate')

            if reliability_warning:
                ax2.set_facecolor('#fff8e1')
                ax2.text(
                    0.5, 0.97,
                    f"\u26a0\ufe0f Only {n_years} years of data \u2014 "
                    f"results need more years to be fully reliable (recommend \u2265{TREND_WARN_YEARS} years)",
                    transform=ax2.transAxes, ha='center', va='top',
                    color='#e65100', fontsize=8.5, style='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e0',
                              edgecolor='#e65100', alpha=0.85)
                )

            sig_label = "\u2705 Trend Confirmed" if p_val < 0.05 else "\u26a0\ufe0f Trend Not Confirmed"
            ax2.set_title(
                f"Yearly Average Moisture: {direction}\n"
                f"{start_date[:4]}\u2013{end_date[:4]}  ({n_years} years) | {sig_label}",
                fontsize=12, fontweight='bold', pad=10
            )
            ax2.set_xlabel("Year", fontsize=12)
            ax2.set_ylabel("Average Soil Moisture (m\u00b3/m\u00b3)", fontsize=12)
            ax2.tick_params(labelsize=11)
            ax2.legend(fontsize=10, loc='best')
            ax2.grid(True, linestyle=':', alpha=0.4)

            plt.savefig(OUTPUT_PATH, dpi=settings['dpi'],
                        bbox_inches='tight', facecolor='white', format='png')
            plt.close()
            print(f"\u2705 Map saved \u2192 {OUTPUT_PATH}")
            return True
        except Exception as e:
            print(f"\u26a0\ufe0f  Visualization error (slope): {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')
            return False


    def _visualize_slope_n_panel(self, clipped_list, var_name, labels,
                                  region_name, display_region):
        """
        Render N slope periods into ONE combined image.
        """
        try:
            n = len(clipped_list)
            if n == 0:
                print("\u26a0\ufe0f  No data to visualize.")
                return False

            ncols = 2
            nrows = n
            fig_w = 18
            fig_h = 7.0 * nrows

            fig, axes = plt.subplots(nrows, ncols,
                                      figsize=(fig_w, fig_h),
                                      constrained_layout=True,
                                      squeeze=False)

            trend_maps = []
            pval_maps  = []
            ts_data    = []

            for clipped in clipped_list:
                annual_da  = _annual_mean(clipped[var_name])
                sp, pv     = _compute_spatial_slope_and_pval(annual_da)
                trend_maps.append(sp)
                pval_maps.append(pv)

                ts_annual  = clipped[var_name].mean(dim=['x', 'y']).groupby('time.year').mean()
                years      = ts_annual.year.values
                y_vals     = ts_annual.values
                x_idx      = np.arange(len(years))
                mask       = ~np.isnan(y_vals)
                n_valid    = int(mask.sum())

                if n_valid >= MIN_TREND_YEARS:
                    sens_val           = _sens_slope(y_vals[mask])
                    mk_p, mk_tau_val   = _mann_kendall_pval_tau(y_vals[mask])
                    ols_s, ols_int, *_ = stats.linregress(x_idx[mask], y_vals[mask])
                else:
                    sens_val = 0.0
                    mk_p     = 1.0
                    mk_tau_val = 0.0
                    ols_s    = 0.0
                    ols_int  = float(np.nanmean(y_vals)) if len(y_vals) else 0.0

                reliability_warning = n_valid < TREND_WARN_YEARS
                ts_data.append((years, y_vals, sens_val, mk_tau_val, mk_p,
                                ols_s, ols_int, x_idx, reliability_warning))

            all_vals = []
            for sp in trend_maps:
                arr = sp.values.ravel()
                arr = arr[~np.isnan(arr)]
                if len(arr):
                    all_vals.extend([float(arr.min()), float(arr.max())])

            if not all_vals:
                print("⚠️  All trend data is NaN — cannot render.")
                plt.close('all')
                return False

            vmin, vmax = -TREND_VMAX, TREND_VMAX
            if vmin == vmax:
                vmin -= 1e-9
                vmax += 1e-9

            map_kwargs = dict(
                x='x' if 'x' in trend_maps[0].coords else 'lon', y='y' if 'y' in trend_maps[0].coords else 'lat', cmap=_TREND_CMAP, center=0,
                vmin=vmin, vmax=vmax,
                cbar_kwargs={'label': 'Rate of Change (m\u00b3/m\u00b3/year)'},
                add_colorbar=True,
            )

            for i, (sp, pv,
                    (years, y_vals, sens_val, mk_tau_val, mk_p,
                     ols_s, ols_int, x_idx, reliability_warning),
                    lbl) in enumerate(
                zip(trend_maps, pval_maps, ts_data, labels)
            ):
                ax_map   = axes[i][0]
                ax_graph = axes[i][1]
                flat_lbl = lbl.replace('\n', '  |  ')
                n_years_row = int((~np.isnan(y_vals)).sum())

                try:
                    sp.plot(ax=ax_map, **map_kwargs)
                except Exception as plot_err:
                    print(f"\u26a0\ufe0f  Row {i+1} spatial plot error: {plot_err}")
                    ax_map.text(0.5, 0.5, f"No data\n{flat_lbl}",
                                ha='center', va='center',
                                transform=ax_map.transAxes)

                _add_stippling(ax_map, sp, pv, dot_size=3)
                _draw_borders(ax_map, self.gdf, region_name)
                
                ax_map.set_title(
                    f"Soil Moisture Trend Map \u2014 {flat_lbl}\n"
                    f"Blue = getting wetter  |  Red = getting drier  |  Dots = confirmed trend",
                    fontsize=11, fontweight='bold', pad=5
                )
                ax_map.set_aspect('equal')
                ax_map.set_xlabel("Longitude", fontsize=10)
                ax_map.set_ylabel("Latitude",  fontsize=10)
                ax_map.tick_params(labelsize=9)

                if len(years) > 0:
                    ax_graph.plot(years, y_vals, 'o-',
                                  color='#2980b9', linewidth=1.8, markersize=5,
                                  alpha=0.85, label='Yearly Moisture')
                    valid_idx = np.where(~np.isnan(y_vals))[0]
                    fi = valid_idx[0] if len(valid_idx) else 0
                    fv = y_vals[fi] if len(valid_idx) else 0.0
                    sens_line = fv + sens_val * (x_idx - fi)
                    ax_graph.plot(years, sens_line,
                                  color='#27ae60', linewidth=2.2, linestyle='-',
                                  label='Main Trend Line')
                    ax_graph.plot(years, ols_int + ols_s * x_idx,
                                  color='#e74c3c', linewidth=1.3, linestyle='--',
                                  alpha=0.55, label='Alternative Estimate')

                if reliability_warning:
                    ax_graph.set_facecolor('#fff8e1')
                    ax_graph.text(
                        0.5, 0.97,
                        f"\u26a0\ufe0f Only {n_years_row} years of data \u2014 needs more years to be fully reliable",
                        transform=ax_graph.transAxes, ha='center', va='top',
                        color='#e65100', fontsize=8, style='italic',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff3e0',
                                  edgecolor='#e65100', alpha=0.8)
                    )

                direction = "\U0001f4c8 Getting Wetter" if sens_val > 0 else "\U0001f4c9 Getting Drier"
                sig_lbl = "\u2705 Trend Confirmed" if mk_p < 0.05 else "\u26a0\ufe0f Trend Not Confirmed"
                ax_graph.set_title(
                    f"Yearly Average Moisture \u2014 {flat_lbl}\n"
                    f"{direction} | {sig_lbl} ({n_years_row} years)",
                    fontsize=11, fontweight='bold', pad=5
                )
                ax_graph.set_xlabel("Year", fontsize=10)
                ax_graph.set_ylabel("Average Soil Moisture (m\u00b3/m\u00b3)", fontsize=10)
                ax_graph.tick_params(labelsize=9)
                ax_graph.legend(fontsize=8, loc='best')
                ax_graph.grid(True, linestyle=':', alpha=0.3)

            fig.suptitle(
                f"Soil Moisture Trend Analysis (Annual Means) \u2014 {display_region}  "
                f"({n} period{'s' if n > 1 else ''})",
                fontsize=15, fontweight='bold', y=1.01
            )
            plt.tight_layout()
            plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
                        facecolor='white', format='png')
            plt.close()
            print(f"\u2705 Combined slope map ({n} period{'s' if n > 1 else ''}) "
                  f"saved \u2192 {OUTPUT_PATH}")
            return True

        except Exception as e:
            print(f"\u26a0\ufe0f  Visualization error (_visualize_slope_n_panel): {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')
            return False

    def visualize_trend_map_only(self, spatial_slope, pval, start_year, end_year, display_region, raw_region):
        """
        Renders ONLY the spatial trend map (no time-series graph) with the reference style.
        Returns the path to the saved image.
        """
        try:
            from matplotlib.colors import LinearSegmentedColormap
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            import matplotlib.gridspec as gridspec

            # _TREND_CMAP is defined globally

            slope_vals  = spatial_slope.values.ravel()
            slope_vals  = slope_vals[~np.isnan(slope_vals)]
            
            wet_pct  = float(np.sum(slope_vals > 0) / len(slope_vals) * 100) if len(slope_vals) else 0
            dry_pct  = 100.0 - wet_pct
            sig_arr  = pval.values.ravel()
            sig_arr  = sig_arr[~np.isnan(sig_arr)]
            sig_pct  = float(np.sum(sig_arr < 0.05) / len(sig_arr) * 100) if len(sig_arr) else 0
            sl_min   = float(np.nanmin(slope_vals)) if len(slope_vals) else 0.0
            sl_max   = float(np.nanmax(slope_vals)) if len(slope_vals) else 0.0
            abs_max  = TREND_VMAX

            fig = plt.figure(figsize=(10, 11), facecolor='white')
            gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[10, 0.8], hspace=0.15)
            
            ax_map   = fig.add_subplot(gs[0])
            ax_stats = fig.add_subplot(gs[1])

            # Map Panel
            spatial_slope.plot(
                ax=ax_map, x='lon' if 'lon' in spatial_slope.coords else 'x', y='lat' if 'lat' in spatial_slope.coords else 'y',
                cmap=_TREND_CMAP, center=0,
                vmin=-abs_max, vmax=abs_max,
                cbar_kwargs={
                    "label"   : "SM Trend (m³/m³ yr⁻¹)",
                    "shrink"  : 0.85,
                    "pad"     : 0.02,
                    "extend"  : "both",
                    "fraction": 0.046,
                },
                add_colorbar=True,
            )
            _add_stippling(ax_map, spatial_slope, pval, dot_size=3)
            _draw_borders(ax_map, self.gdf, raw_region)

            ax_map.set_title(
                f"Soil Moisture Trend: {display_region}  ({start_year}–{end_year})",
                fontsize=15, fontweight='bold', pad=10, loc='center',
                fontfamily='DejaVu Sans'
            )
            ax_map.set_title(
                f"Pixel-wise OLS on SM  |  Stippling (•) = significant pixels (p < 0.05)",
                fontsize=9.5, pad=2, loc='center', color='#444444',
                style='italic', y=-0.03
            )
            ax_map.set_xlabel("Longitude (°E)", fontsize=11)
            ax_map.set_ylabel("Latitude (°N)",  fontsize=11)
            ax_map.tick_params(labelsize=10)
            ax_map.set_facecolor('white')
            for sp in ax_map.spines.values():
                sp.set_edgecolor('#cccccc')

            _legend_elements = [
                Patch(facecolor='#1a7d73', edgecolor='none', label='Wetting trend (+)'),
                Patch(facecolor='#a0522d', edgecolor='none', label='Drying trend (−)'),
                Line2D([0], [0], marker='.', color='none', markerfacecolor='black', markersize=6, label='Significant (p < 0.05)'),
            ]
            ax_map.legend(
                handles=_legend_elements, loc='lower left', title='Trend  |  Significance',
                title_fontsize=8.5, fontsize=8.5, framealpha=0.88, edgecolor='#999999', fancybox=False,
            )

            # Stats Panel
            ax_stats.axis('off')
            ax_stats.set_facecolor('#f0f4f4')
            for sp in ax_stats.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor('#cccccc')
            
            stats_text1 = (
                f"Period: {start_year}–{end_year}  │  "
                f"Wetting pixels: {wet_pct:.1f}%  │  "
                f"Drying pixels: {dry_pct:.1f}%"
            )
            stats_text2 = (
                f"Significant pixels (p < 0.05): {sig_pct:.1f}%  │  "
                f"Trend range: {sl_min:.4f} – {sl_max:.4f} m³/m³/yr"
            )
            ax_stats.text(0.5, 0.70, stats_text1, ha='center', va='center', fontsize=9.5, transform=ax_stats.transAxes, color='#1a5276')
            ax_stats.text(0.5, 0.22, stats_text2, ha='center', va='center', fontsize=9.0, transform=ax_stats.transAxes, color='#1a5276', style='italic')

            fig.patch.set_facecolor('white')
            out_path = OUTPUT_PATH.replace('.png', '_map_only.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
            plt.close()
            return out_path
        except Exception as e:
            import traceback
            print(f"⚠️  Visualization error (map only): {e}")
            traceback.print_exc()
            plt.close('all')
            return None

    def _analyze_minimum(self, clipped, var_name, display_region, raw_region,
                         start_date, end_date, output_type):
        data  = clipped[var_name]
        total = int(data.size)
        null  = int(data.isnull().sum())

        expected_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
        spatial_size = total // max(1, data.time.size) if hasattr(data, 'time') and data.time.size > 0 else 0
        expected_total = expected_days * spatial_size
        
        missing_days_count = max(0, expected_days - (data.time.size if hasattr(data, 'time') else 1))
        true_null = null + (missing_days_count * spatial_size)
        true_total = expected_total if expected_total > 0 else total

        stats_dict = {
            'min':         float(data.min()),
            'count':       int(data.count()),
            'missing_pct': (true_null / true_total * 100) if true_total else 0
        }
        output = OutputFormatter.format_minimum(
            stats_dict, output_type, display_region, start_date, end_date)

        viz_created = False
        if output_type in ['map', 'both']:
            viz_created = self._visualize_minimum(
                clipped, var_name, display_region, raw_region,
                start_date, end_date)

        return output, viz_created

    def _visualize_minimum(self, clipped, var_name, display_region, raw_region,
                           start_date, end_date):
        try:
            settings = FIGURE_SETTINGS['minimum']
            fig, ax  = plt.subplots(1, 1, figsize=settings['figsize'],
                                    constrained_layout=True)
            min_data = clipped[var_name].min(dim='time')
            min_data.plot(
                ax=ax, x='x', y='y', cmap='RdYlGn',
                cbar_kwargs={'label': 'Minimum Soil Moisture (m³/m³)', 'shrink': 0.85})
            _draw_borders(ax, self.gdf, raw_region)
            ax.set_title(f"Minimum Soil Moisture: {display_region}\n"
                         f"({start_date} to {end_date})",
                         fontsize=16, fontweight='bold', pad=10)
            ax.set_xlabel("Longitude", fontsize=13)
            ax.set_ylabel("Latitude",  fontsize=13)
            ax.tick_params(labelsize=11)
            ax.set_aspect('equal')
            plt.savefig(OUTPUT_PATH, dpi=settings['dpi'], bbox_inches='tight', facecolor='white', format='png')
            plt.close()
            print(f"✅ Map saved → {OUTPUT_PATH}")
            return True
        except Exception as e:
            print(f"⚠️  Visualization error (minimum): {e}")
            traceback.print_exc()
            plt.close('all')
            return False

    def _analyze_maximum(self, clipped, var_name, display_region, raw_region,
                         start_date, end_date, output_type):
        data  = clipped[var_name]
        total = int(data.size)
        null  = int(data.isnull().sum())

        expected_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
        spatial_size = total // max(1, data.time.size) if hasattr(data, 'time') and data.time.size > 0 else 0
        expected_total = expected_days * spatial_size
        
        missing_days_count = max(0, expected_days - (data.time.size if hasattr(data, 'time') else 1))
        true_null = null + (missing_days_count * spatial_size)
        true_total = expected_total if expected_total > 0 else total

        stats_dict = {
            'max':         float(data.max()),
            'count':       int(data.count()),
            'missing_pct': (true_null / true_total * 100) if true_total else 0
        }
        output = OutputFormatter.format_maximum(
            stats_dict, output_type, display_region, start_date, end_date)

        viz_created = False
        if output_type in ['map', 'both']:
            viz_created = self._visualize_maximum(
                clipped, var_name, display_region, raw_region,
                start_date, end_date)

        return output, viz_created

    def _visualize_maximum(self, clipped, var_name, display_region, raw_region,
                           start_date, end_date):
        try:
            settings = FIGURE_SETTINGS['maximum']
            fig, ax  = plt.subplots(1, 1, figsize=settings['figsize'],
                                    constrained_layout=True)
            max_data = clipped[var_name].max(dim='time')
            max_data.plot(
                ax=ax, x='x', y='y', cmap='Blues',
                cbar_kwargs={'label': 'Maximum Soil Moisture (m³/m³)', 'shrink': 0.85})
            _draw_borders(ax, self.gdf, raw_region)
            ax.set_title(f"Maximum Soil Moisture: {display_region}\n"
                         f"({start_date} to {end_date})",
                         fontsize=16, fontweight='bold', pad=10)
            ax.set_xlabel("Longitude", fontsize=13)
            ax.set_ylabel("Latitude",  fontsize=13)
            ax.tick_params(labelsize=11)
            ax.set_aspect('equal')
            plt.savefig(OUTPUT_PATH, dpi=settings['dpi'], bbox_inches='tight', facecolor='white', format='png')
            plt.close()
            print(f"✅ Map saved → {OUTPUT_PATH}")
            return True
        except Exception as e:
            print(f"⚠️  Visualization error (maximum): {e}")
            traceback.print_exc()
            plt.close('all')
            return False

    # ================================================================== #
    # COMPARISON HANDLER  (routes to N-way time or region comparison)
    # ================================================================== #

    def _handle_comparison(self, region, start_date, end_date,
                            output_type, comparison_info):
        if not comparison_info:
            return "❌ Comparison info not provided.", False

        comp_type   = comparison_info.get('comparison_type', 'time')
        comp_metric = comparison_info.get('comparison_metric', 'mean')

        if comp_type == 'region':
            return self._analyze_comparison_region(
                region,
                comparison_info.get('comparison_region2', ''),
                start_date, end_date,
                output_type, comp_metric
            )
        else:
            # N-way time comparison
            periods = comparison_info.get('comparison_periods', [])

            # Back-compat: if old keys used, reconstruct periods list
            if not periods:
                p1 = comparison_info.get('comparison_period1')
                p2 = comparison_info.get('comparison_period2')
                if p1 and p2:
                    periods = [p1, p2]

            if len(periods) < 2:
                return (
                    "❌ Two or more time periods are required for time comparison. "
                    "Please specify all periods clearly."
                ), False

            return self._analyze_comparison_n_periods(
                region, periods, output_type, comp_metric
            )

    # ── METRIC HELPERS ───────────────────────────────────────────────────

    @staticmethod
    def _compute_metric(data_array, metric: str) -> float:
        if metric == 'min':
            return float(data_array.min())
        elif metric == 'max':
            return float(data_array.max())
        elif metric == 'slope':
            ts    = data_array.mean(dim=['x', 'y'])
            mask  = ~np.isnan(ts)
            clean = ts.where(mask, drop=True)
            if len(clean) < 3:
                return float('nan')
            slope, *_ = stats.linregress(np.arange(len(clean)), clean.values)
            return float(slope)
        else:
            return float(data_array.mean())

    @staticmethod
    def _metric_map(data_array, metric: str):
        if metric == 'min':
            return data_array.min(dim='time')
        elif metric == 'max':
            return data_array.max(dim='time')
        elif metric == 'slope':
            return _compute_spatial_slope(data_array)
        else:
            return data_array.mean(dim='time')

    # ── N-WAY TIME COMPARISON ────────────────────────────────────────────

    def _analyze_comparison_n_periods(self, region, periods, output_type, comp_metric):
        """
        Compare the same region across N time periods.
        For 'slope' metric: ONE combined image with spatial map + trend graph per period.
        For other metrics: ONE combined image via _visualize_n_panel.
        """
        v = list(self.ds.data_vars)[0]

        period_data    = []   # list of clipped DataArrays (one per period)
        clipped_ds_list = []  # list of clipped Datasets — kept for slope visualizer
        period_labels  = []
        display_region = region.title()

        for idx, (s, e) in enumerate(periods):
            if len(s) == 7: s += "-01"
            if len(e) == 7: e += "-28"

            subset = self.ds.sel(time=slice(s, e)).compute()
            clipped, disp, ok = self._clip_region(subset, region)
            if not ok:
                return clipped, False

            if clipped[v].size == 0 or clipped[v].isnull().all():
                return (
                    f"❌ No valid data for {disp} in period {s} to {e}."
                ), False

            display_region = disp
            period_data.append(clipped[v])      # DataArray
            clipped_ds_list.append(clipped)     # Dataset (for _visualize_slope_n_panel)
            label = f"Period {idx+1}\n{s} to {e}"
            period_labels.append(label)

        values = [self._compute_metric(da, comp_metric) for da in period_data]

        output = OutputFormatter.format_comparison_n_periods(
            values        = values,
            period_labels = period_labels,
            region        = display_region,
            metric        = comp_metric,
            output_type   = output_type,
        )

        viz_created = False
        if output_type in ['map', 'both']:
            if comp_metric == 'slope':
                # _visualize_slope_n_panel renders both spatial slope map AND
                # temporal trend graph for every period in one combined image.
                viz_created = self._visualize_slope_n_panel(
                    clipped_list   = clipped_ds_list,
                    var_name       = v,
                    labels         = period_labels,
                    region_name    = region,
                    display_region = display_region,
                )
            else:
                maps = [self._metric_map(da, comp_metric) for da in period_data]

                # ── Add difference map when exactly 2 periods are compared ──
                diff_index = None
                if len(maps) == 2:
                    try:
                        diff_map = maps[1] - maps[0]
                        maps.append(diff_map)
                        period_labels.append(
                            f"Difference\n(Period 2 \u2212 Period 1)"
                        )
                        diff_index = 2   # third panel (0-based)
                    except Exception as diff_err:
                        print(f"\u26a0\ufe0f  Could not compute difference map: {diff_err}")

                viz_created = self._visualize_n_panel(
                    maps         = maps,
                    labels       = period_labels,
                    region_name  = region,
                    region_names = None,
                    suptitle     = (
                        f"{len(periods)}-Period Comparison: {display_region}  |  "
                        f"Metric: {comp_metric.upper()}"
                    ),
                    metric       = comp_metric,
                    diff_index   = diff_index,
                )

        return output, viz_created

    # ── REGION COMPARISON ────────────────────────────────────────────────

    def _analyze_comparison_region(self, region1, region2,
                                    start_date, end_date,
                                    output_type, comp_metric):
        """
        Compare two regions for the same time period.
        Always produces ONE combined image via _visualize_n_panel.
        """
        if not region2:
            return "❌ Second region not found. Please name two regions.", False

        subset = self.ds.sel(time=slice(start_date, end_date)).compute()

        clipped1, display1, ok = self._clip_region(subset, region1)
        if not ok:
            return clipped1, False

        clipped2, display2, ok = self._clip_region(subset, region2)
        if not ok:
            return clipped2, False

        v = list(self.ds.data_vars)[0]

        if clipped1[v].size == 0:
            return f"❌ No spatial data found for {display1} in this period.", False
        if clipped2[v].size == 0:
            return f"❌ No spatial data found for {display2} in this period.", False
        if clipped1[v].isnull().all():
            return f"❌ All values missing for {display1}.", False
        if clipped2[v].isnull().all():
            return f"❌ All values missing for {display2}.", False

        value1 = self._compute_metric(clipped1[v], comp_metric)
        value2 = self._compute_metric(clipped2[v], comp_metric)

        stats_dict = {'value1': value1, 'value2': value2}

        output = OutputFormatter.format_comparison_region(
            stats_dict, output_type, display1, display2,
            start_date, end_date, comp_metric
        )

        viz_created = False
        if output_type in ['map', 'both']:
            map1 = self._metric_map(clipped1[v], comp_metric)
            map2 = self._metric_map(clipped2[v], comp_metric)
            viz_created = self._visualize_n_panel(
                maps         = [map1, map2],
                labels       = [display1, display2],
                region_name  = None,
                region_names = [region1, region2],
                suptitle     = (
                    f"Regional Comparison: {display1} vs {display2}  |  "
                    f"{start_date} to {end_date}  |  "
                    f"Metric: {comp_metric.upper()}"
                ),
                metric       = comp_metric,
            )

        return output, viz_created

    # ── UNIFIED N-PANEL VISUALIZATION ────────────────────────────────────

    def _visualize_n_panel(
        self,
        maps,
        labels,
        region_name,
        suptitle,
        metric,
        region_names=None,
        diff_index=None,
    ):
        """
        Render a clean N-panel comparison map in a dynamic grid layout.
        ALWAYS saves to OUTPUT_PATH (latest_analysis.png) as ONE image.

        Layout rules:
          N=1  → 1×1
          N=2  → 1×2
          N=3  → 1×3
          N=4  → 2×2
          N=5  → 2×3  (one empty cell)
          N=6  → 2×3
          N=7+ → rows × 3  grid

        diff_index : int or None
            When set, the panel at this index is treated as a difference map
            and rendered with its own RdBu colorbar independent of the others.
        """
        try:
            n = len(maps)
            if n == 0:
                print("⚠️  No maps to visualize.")
                return False

            ncols = min(n, 3)
            nrows = math.ceil(n / ncols)

            fig_w = ncols * 7.0
            fig_h = nrows * 6.5

            fig, axes = plt.subplots(
                nrows, ncols, figsize=(fig_w, fig_h), squeeze=False
            )

            # ── Shared colorscale for data panels (excluding diff panel) ──────
            data_indices = [i for i in range(n) if i != diff_index]
            valid_vals = []
            for i in data_indices:
                arr = maps[i].values.ravel()
                arr = arr[~np.isnan(arr)]
                if len(arr):
                    valid_vals.extend([float(arr.min()), float(arr.max())])

            if not valid_vals:
                print("⚠️  All map data is NaN — cannot render.")
                plt.close('all')
                return False

            g_min = min(valid_vals)
            g_max = max(valid_vals)

            if metric == 'slope':
                cmap     = 'RdBu'
                cb_label = 'Slope (m³/m³/day)'
                center   = 0.0
                abs_max  = max(abs(g_min), abs(g_max))
                vmin, vmax = -abs_max, abs_max
            elif metric == 'min':
                cmap     = 'RdYlGn'
                cb_label = 'Minimum Soil Moisture (m³/m³)'
                center   = None
                vmin, vmax = g_min, g_max
            elif metric == 'max':
                cmap     = 'Blues'
                cb_label = 'Maximum Soil Moisture (m³/m³)'
                center   = None
                vmin, vmax = g_min, g_max
            else:
                cmap     = 'YlGnBu'
                cb_label = 'Mean Soil Moisture (m³/m³)'
                center   = None
                vmin, vmax = g_min, g_max

            if vmin == vmax:
                vmin -= 1e-6
                vmax += 1e-6

            # ── Difference panel colorscale ───────────────────────────────────
            diff_plot_kwargs = None
            if diff_index is not None:
                diff_arr = maps[diff_index].values.ravel()
                diff_arr = diff_arr[~np.isnan(diff_arr)]
                if len(diff_arr):
                    abs_diff = max(abs(float(diff_arr.min())), abs(float(diff_arr.max())))
                    if abs_diff == 0:
                        abs_diff = 1e-6
                else:
                    abs_diff = 1e-6
                diff_plot_kwargs = dict(
                    x='x', y='y', cmap='RdBu_r',
                    vmin=-abs_diff, vmax=abs_diff,
                    center=0.0,
                    cbar_kwargs={'label': 'Difference (m³/m³)'},
                    add_colorbar=True,
                )

            data_plot_kwargs = dict(
                x='x', y='y', cmap=cmap,
                vmin=vmin, vmax=vmax,
                cbar_kwargs={'label': cb_label},
                add_colorbar=True,
            )
            if center is not None:
                data_plot_kwargs['center'] = center

            for i, (m, lbl) in enumerate(zip(maps, labels)):
                row, col = divmod(i, ncols)
                ax = axes[row][col]

                # Choose correct plot kwargs for this panel
                pk = diff_plot_kwargs if (i == diff_index and diff_plot_kwargs is not None) \
                     else data_plot_kwargs

                try:
                    m.plot(ax=ax, **pk)
                except Exception as plot_err:
                    print(f"⚠️  Panel {i+1} plot error: {plot_err}")
                    ax.text(0.5, 0.5, f"No data\n{lbl}",
                            ha='center', va='center', transform=ax.transAxes)

                rname = (region_names[i] if region_names else region_name)
                _draw_borders(ax, self.gdf, rname)

                # Highlight the difference panel title
                title_color = '#8B0000' if i == diff_index else 'black'
                clean_lbl = lbl.replace('\n', '  |  ')
                ax.set_title(clean_lbl, fontsize=12, fontweight='bold',
                             pad=6, color=title_color)
                ax.set_aspect('equal')
                ax.set_xlabel("Longitude", fontsize=10)
                ax.set_ylabel("Latitude",  fontsize=10)
                ax.tick_params(labelsize=9)

            total_cells = nrows * ncols
            for j in range(n, total_cells):
                row, col = divmod(j, ncols)
                axes[row][col].set_visible(False)

            fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)
            plt.tight_layout()
            plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
                        facecolor='white', format='png')
            plt.close()
            print(f"✅ Combined comparison map ({n} panels) saved → {OUTPUT_PATH}")
            return True

        except Exception as e:
            print(f"⚠️  Visualization error (_visualize_n_panel): {e}")
            traceback.print_exc()
            plt.close('all')
            return False