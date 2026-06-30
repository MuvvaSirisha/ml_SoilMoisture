"""
utils.py
========
Helper functions for validation, formatting, and missing data analysis.
"""

import re
from datetime import datetime, timedelta
import numpy as np


class OutputFormatter:
    """Formats analysis results based on output type and operation."""

    # ------------------------------------------------------------------
    # MEAN
    # ------------------------------------------------------------------

    @staticmethod
    def format_mean(data, output_type, region, start_date, end_date):
        if output_type == 'scalar':
            return OutputFormatter._format_scalar_mean(data, region, start_date, end_date)
        elif output_type == 'map':
            return ""
        else:
            return OutputFormatter._format_scalar_mean(data, region, start_date, end_date)

    @staticmethod
    def _format_scalar_mean(data, region, start_date, end_date):
        is_single_day = (start_date == end_date)
        duration = (datetime.strptime(end_date, '%Y-%m-%d') -
                    datetime.strptime(start_date, '%Y-%m-%d')).days + 1

        if is_single_day:
            return (
                f"**\U0001f4ca Soil Moisture — {region}**\n\n"
                f"**Date:** {start_date}\n\n"
                f"---\n\n"
                f"**Soil Moisture:** `{data['mean']:.6f} m\u00b3/m\u00b3`"
            )
        return (
            f"**\U0001f4ca Mean Soil Moisture — {region}**\n\n"
            f"**Period:** {start_date} \u2192 {end_date} &nbsp; ({duration} days)\n\n"
            f"---\n\n"
            f"**Mean Moisture:** `{data['mean']:.6f} m\u00b3/m\u00b3`"
        )


    # ------------------------------------------------------------------
    # SLOPE
    # ------------------------------------------------------------------

    @staticmethod
    def format_slope(data, output_type, region, start_date, end_date):
        if output_type == 'scalar':
            return OutputFormatter._format_scalar_slope(data, region, start_date, end_date)
        elif output_type == 'map':
            return ""
        else:
            return OutputFormatter._format_scalar_slope(data, region, start_date, end_date)

    @staticmethod
    def _format_scalar_slope(data, region, start_date, end_date):
        from Config import TREND_WARN_YEARS
        n_years      = data.get('n_years', data.get('count', '?'))
        mk_tau       = data.get('mk_tau', 0.0)
        slope        = data.get('slope', 0.0)
        p_val        = data.get('p_value', 1.0)
        total_change = data.get('total_change', slope * (n_years if isinstance(n_years, (int, float)) else 0))
        direction    = "📈 Increasing (Getting Wetter)" if slope > 0 else "📉 Decreasing (Getting Drier)"
        significance = "✅ Statistically Significant" if p_val < 0.05 else "⚠️ Not Significant (MK p ≥ 0.05)"
        reliability_warning = data.get('reliability_warning', False)

        warn_block = ""
        if reliability_warning:
            warn_block = (
                f"\n\n> ⚠️ **Caution — Short Time Series ({n_years} years)**\n"
                f"> We recommend at least {TREND_WARN_YEARS} years of data for a reliable trend.\n"
                f"> These results are indicative only. Consider extending your date range."
            )

        return (
            f"**📈 Soil Moisture Trend — {region}**\n\n"
            f"**Period:** {start_date} → {end_date} &nbsp; ({n_years} years)\n\n"
            f"---\n\n"
            f"**Method:** Robust trend analysis (minimizes the effect of extreme years)\n\n"
            f"**Overall Trend:** {direction} &nbsp; {significance}\n\n"
            f"**Estimated Rate:** `{slope:.8f} m³/m³/year`\n\n"
            f"**Total Change:** `{total_change:+.6f} m³/m³` over {n_years} years\n\n"
            f"> Positive rate = soil getting wetter &nbsp;|&nbsp; Negative rate = soil getting drier\n\n"
            f"> Stippled pixels on map = Confirmed trend &nbsp;|&nbsp; "
            f"Green line = Main trend &nbsp;|&nbsp; Red dashed = Alternative estimate"
            f"{warn_block}"
        )


    # ------------------------------------------------------------------
    # MINIMUM
    # ------------------------------------------------------------------

    @staticmethod
    def format_minimum(data, output_type, region, start_date, end_date):
        if output_type == 'scalar':
            return OutputFormatter._format_scalar_minimum(data, region, start_date, end_date)
        elif output_type == 'map':
            return ""
        else:
            return OutputFormatter._format_scalar_minimum(data, region, start_date, end_date)

    @staticmethod
    def _format_scalar_minimum(data, region, start_date, end_date):
        duration = (datetime.strptime(end_date, '%Y-%m-%d') -
                    datetime.strptime(start_date, '%Y-%m-%d')).days + 1
        return (
            f"**\U0001f4c9 Minimum Soil Moisture — {region}**\n\n"
            f"**Period:** {start_date} \u2192 {end_date} &nbsp; ({duration} days)\n\n"
            f"---\n\n"
            f"**Minimum Moisture:** `{data['min']:.6f} m\u00b3/m\u00b3` &nbsp; \u26a0\ufe0f Driest value"
        )

    # ------------------------------------------------------------------
    # MAXIMUM
    # ------------------------------------------------------------------

    @staticmethod
    def format_maximum(data, output_type, region, start_date, end_date):
        if output_type == 'scalar':
            return OutputFormatter._format_scalar_maximum(data, region, start_date, end_date)
        elif output_type == 'map':
            return ""
        else:
            return OutputFormatter._format_scalar_maximum(data, region, start_date, end_date)

    @staticmethod
    def _format_scalar_maximum(data, region, start_date, end_date):
        duration = (datetime.strptime(end_date, '%Y-%m-%d') -
                    datetime.strptime(start_date, '%Y-%m-%d')).days + 1
        return (
            f"**\U0001f4c8 Maximum Soil Moisture — {region}**\n\n"
            f"**Period:** {start_date} \u2192 {end_date} &nbsp; ({duration} days)\n\n"
            f"---\n\n"
            f"**Maximum Moisture:** `{data['max']:.6f} m\u00b3/m\u00b3` &nbsp; \u2b50 Wettest value"
        )

    # ==================================================================
    # COMPARISON — SHARED WINNER LOGIC
    # ==================================================================

    @staticmethod
    def _pick_winner(value1, value2, label1, label2, metric):
        """Winner logic for 2-way comparison (back-compat)."""
        metric = metric.lower()

        if metric == 'min':
            if value1 <= value2:
                summary = f"{label1} is DRIER  ⚠️  (lower minimum: {value1:.6f} m³/m³)"
                m1, m2  = "⚠️  DRIEST", ""
            else:
                summary = f"{label2} is DRIER  ⚠️  (lower minimum: {value2:.6f} m³/m³)"
                m1, m2  = "", "⚠️  DRIEST"

        elif metric == 'max':
            if value1 >= value2:
                summary = f"{label1} has the WETTEST PEAK  ⭐ (max: {value1:.6f} m³/m³)"
                m1, m2  = "⭐ HIGHEST PEAK", ""
            else:
                summary = f"{label2} has the WETTEST PEAK  ⭐ (max: {value2:.6f} m³/m³)"
                m1, m2  = "", "⭐ HIGHEST PEAK"

        elif metric == 'slope':
            if abs(value1) >= abs(value2):
                direction = "WETTER" if value1 > 0 else "DRIER"
                summary = (f"{label1} has STRONGER TREND  📈 "
                           f"(slope: {value1:+.8f}, getting {direction})")
                m1, m2  = "📈 STRONGER TREND", ""
            else:
                direction = "WETTER" if value2 > 0 else "DRIER"
                summary = (f"{label2} has STRONGER TREND  📈 "
                           f"(slope: {value2:+.8f}, getting {direction})")
                m1, m2  = "", "📈 STRONGER TREND"

        else:   # mean
            if value1 >= value2:
                summary = (f"{label1} is WETTER ON AVERAGE  💧 "
                           f"(mean: {value1:.6f} m³/m³)")
                m1, m2  = "💧 HIGHER MEAN", ""
            else:
                summary = (f"{label2} is WETTER ON AVERAGE  💧 "
                           f"(mean: {value2:.6f} m³/m³)")
                m1, m2  = "", "💧 HIGHER MEAN"

        return summary, m1, m2

    @staticmethod
    def _pick_winner_n(values, labels, metric):
        """
        Winner logic for N-way comparison.

        Returns
        -------
        winner_idx     : int   — index of the winning period/region
        winner_label   : str   — label of the winner
        winner_value   : float — metric value of the winner
        winner_summary : str   — one-line human-readable result
        badges         : list  — badge string per entry ("" or badge text)
        """
        metric = metric.lower()

        if metric == 'min':
            winner_idx  = int(np.nanargmin(values))
            badge_text  = "⚠️  DRIEST"
            summary_tpl = "{label} is DRIEST  ⚠️  (lowest minimum: {val:.6f} m³/m³)"
        elif metric == 'max':
            winner_idx  = int(np.nanargmax(values))
            badge_text  = "⭐ HIGHEST PEAK"
            summary_tpl = "{label} has the WETTEST PEAK  ⭐ (max: {val:.6f} m³/m³)"
        elif metric == 'slope':
            winner_idx  = int(np.nanargmax([abs(v) for v in values]))
            badge_text  = "📈 STRONGER TREND"
            v_win = values[winner_idx]
            direction   = "WETTER" if v_win > 0 else "DRIER"
            summary_tpl = (
                "{label} has STRONGER TREND  📈 "
                f"(slope: {v_win:+.8f}, getting {direction})"
            )
        else:   # mean
            winner_idx  = int(np.nanargmax(values))
            badge_text  = "💧 HIGHEST MEAN"
            summary_tpl = "{label} is WETTEST ON AVERAGE  💧 (mean: {val:.6f} m³/m³)"

        winner_label = labels[winner_idx]
        winner_value = values[winner_idx]
        badges       = [""] * len(values)
        badges[winner_idx] = badge_text

        # Clean up newlines in label for inline display
        clean_label = winner_label.replace('\n', ' ')
        summary = summary_tpl.format(label=clean_label, val=winner_value)

        return winner_idx, winner_label, winner_value, summary, badges

    # ------------------------------------------------------------------
    # COMPARISON — N TIME PERIODS  (NEW in v2.5)
    # ------------------------------------------------------------------

    @staticmethod
    def format_comparison_n_periods(values, period_labels, region, metric, output_type):
        """
        Format N-period time comparison scalar output.

        Parameters
        ----------
        values        : list of float  — metric value per period
        period_labels : list of str    — label per period (may contain \\n)
        region        : str
        metric        : str  ('mean'|'min'|'max'|'slope')
        output_type   : str  ('scalar'|'map'|'both')
        """
        if output_type == 'map':
            return ""

        scalar_block = OutputFormatter._format_scalar_n_periods(
            values, period_labels, region, metric
        )

        return scalar_block

    @staticmethod
    def _format_scalar_n_periods(values, period_labels, region, metric):
        n = len(values)
        _, _, _, winner_summary, badges = OutputFormatter._pick_winner_n(
            values, period_labels, metric
        )
        unit_map = {'mean': 'm³/m³', 'min': 'm³/m³', 'max': 'm³/m³', 'slope': 'm³/m³/day'}
        unit = unit_map.get(metric, 'm³/m³')

        lines = [f"**🔄 Soil Moisture Comparison — {region}** ({metric.upper()})\n"]
        lines.append("---\n")
        for i, (lbl, val) in enumerate(zip(period_labels, values), 1):
            badge = badges[i - 1]
            flat_lbl = lbl.replace('\n', ' | ')
            val_str = f"`{val:.6f} {unit}`" if not (isinstance(val, float) and np.isnan(val)) else "`N/A`"
            badge_str = f" &nbsp; **{badge}**" if badge else ""
            lines.append(f"**Period {i}:** {flat_lbl}  →  {val_str}{badge_str}\n")
        lines.append(f"\n🏆 **Result:** {winner_summary}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # COMPARISON — 2 TIME PERIODS  (back-compat, delegates to N-period)
    # ------------------------------------------------------------------

    @staticmethod
    def format_comparison_time(data, output_type, region,
                                period1_label, period2_label, metric_label):
        """Back-compat wrapper — delegates to format_comparison_n_periods."""
        values = [data['value1'], data['value2']]
        labels = [f"Period 1\n{period1_label}", f"Period 2\n{period2_label}"]
        return OutputFormatter.format_comparison_n_periods(
            values, labels, region, metric_label, output_type
        )

    # ------------------------------------------------------------------
    # COMPARISON — REGIONS  (unchanged from v2.4)
    # ------------------------------------------------------------------

    @staticmethod
    def format_comparison_region(data, output_type, region1, region2,
                                  start_date, end_date, metric_label):
        if output_type == 'scalar':
            return OutputFormatter._format_scalar_comparison_region(
                data, region1, region2, start_date, end_date, metric_label
            )
        elif output_type == 'map':
            return ""
        else:
            return OutputFormatter._format_scalar_comparison_region(
                data, region1, region2, start_date, end_date, metric_label
            )

    @staticmethod
    def _format_scalar_comparison_region(data, region1, region2,
                                          start_date, end_date, metric_label):
        value1 = data['value1']
        value2 = data['value2']
        winner, m1, m2 = OutputFormatter._pick_winner(
            value1, value2, region1, region2, metric_label
        )
        m1_badge = f" &nbsp; **{m1}**" if m1 else ""
        m2_badge = f" &nbsp; **{m2}**" if m2 else ""
        return (
            f"**🔄 Regional Comparison — {metric_label.upper()}**\n\n"
            f"**Period:** {start_date} → {end_date}\n\n"
            f"---\n\n"
            f"**{region1}:** `{value1:.6f} m³/m³`{m1_badge}\n\n"
            f"**{region2}:** `{value2:.6f} m³/m³`{m2_badge}\n\n"
            f"🏆 **Result:** {winner}"
        )


# ============================================================================
# DATE ANALYSER
# ============================================================================

class DateAnalyzer:
    """Analyzes date ranges for missing data."""

    @staticmethod
    def find_missing_dates(available_dates, start_date, end_date):
        import pandas as pd
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end   = datetime.strptime(end_date,   '%Y-%m-%d')

        expected_dates = set()
        current = start
        while current <= end:
            expected_dates.add(current.date())
            current += timedelta(days=1)

        available_set = set()
        for d in available_dates:
            if isinstance(d, np.datetime64):
                available_set.add(pd.Timestamp(d).date())
            elif isinstance(d, datetime):
                available_set.add(d.date())
            else:
                available_set.add(d)

        missing_dates   = sorted(expected_dates - available_set)
        missing_periods = []

        if missing_dates:
            period_start = missing_dates[0]
            period_end   = missing_dates[0]
            for date in missing_dates[1:]:
                if (date - period_end).days == 1:
                    period_end = date
                else:
                    missing_periods.append((period_start, period_end))
                    period_start = date
                    period_end   = date
            missing_periods.append((period_start, period_end))

        available_min = min(available_set).strftime('%Y-%m-%d') if available_set else None
        available_max = max(available_set).strftime('%Y-%m-%d') if available_set else None

        return {
            'total_expected':     len(expected_dates),
            'total_available':    len(available_set),
            'total_missing':      len(missing_dates),
            'missing_percentage': (
                len(missing_dates) / len(expected_dates) * 100
            ) if expected_dates else 0,
            'missing_dates':      missing_dates[:10],
            'missing_periods':    missing_periods,
            'has_gaps':           len(missing_dates) > 0,
            'available_min':      available_min,
            'available_max':      available_max,
        }

    @staticmethod
    def format_missing_report(missing_info, region, start_date, end_date):
        if missing_info['total_missing'] == 0:
            return f"✅ Complete data for {region} ({start_date} to {end_date})\n"
            
        if missing_info['missing_percentage'] == 100.0:
            msg = f"❌ No data available for {region} ({start_date} to {end_date}).\n"
            if missing_info.get('available_min') and missing_info.get('available_max'):
                msg += f"ℹ️ I have data available from {missing_info['available_min']} to {missing_info['available_max']}.\n"
            return msg

        report = f"""
╔════════════════════════════════════════════════════════════════════════╗
║                    📅 DATA AVAILABILITY REPORT                       ║
╠════════════════════════════════════════════════════════════════════════╣
║ Region: {region}
║ Period: {start_date} to {end_date}
║
║ Total Expected Days:  {missing_info['total_expected']}
║ Total Available Days: {missing_info['total_available']}
║ Missing Days:         {missing_info['total_missing']}
║ Missing %:            {missing_info['missing_percentage']:.1f}%
║
║ Missing Periods:
"""
        for period_start, period_end in missing_info['missing_periods'][:5]:
            if period_start == period_end:
                report += f"║   • {period_start}\n"
            else:
                report += f"║   • {period_start} to {period_end}\n"

        if len(missing_info['missing_periods']) > 5:
            report += f"║   ... and {len(missing_info['missing_periods']) - 5} more periods\n"

        report += "╚════════════════════════════════════════════════════════════════════════╝\n"
        return report


# ============================================================================
# QUERY VALIDATOR
# ============================================================================

class QueryValidator:
    """Validates query classification results."""

    @staticmethod
    def validate_dates(start_date, end_date):
        """
        Validate and if necessary auto-correct date order.

        If start_date > end_date (user entered dates in decreasing order),
        the dates are swapped automatically and a friendly notice is printed.
        The returned dict is always valid=True in that case so processing
        continues uninterrupted.

        Returns
        -------
        dict with keys:
            valid       : bool
            message     : str
            start_date  : str  — corrected start date (YYYY-MM-DD)
            end_date    : str  — corrected end date   (YYYY-MM-DD)
        """
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end   = datetime.strptime(end_date,   '%Y-%m-%d')

            if start > end:
                # ── Auto-swap and inform the user ─────────────────────
                corrected_start = end_date
                corrected_end   = start_date
                print(
                    f"\n📅 Note: Dates were entered in reverse order "
                    f"({start_date} → {end_date}).\n"
                    f"   Interpreting as: {corrected_start} to {corrected_end} "
                    f"and proceeding normally.\n"
                )
                return {
                    'valid':      True,
                    'message':    (
                        f"Date order corrected: {corrected_start} to {corrected_end}"
                    ),
                    'start_date': corrected_start,
                    'end_date':   corrected_end,
                }

            return {
                'valid':      True,
                'message':    'Valid dates',
                'start_date': start_date,
                'end_date':   end_date,
            }

        except ValueError:
            return {
                'valid':      False,
                'message':    "Invalid date format. Use YYYY-MM-DD or Month Year",
                'start_date': start_date,
                'end_date':   end_date,
            }

    @staticmethod
    def validate_operation(operation):
        valid_ops = ['mean', 'slope', 'minimum', 'maximum', 'comparison']
        if operation not in valid_ops:
            return {'valid': False, 'message': f"Unknown operation: {operation}"}
        return {'valid': True, 'message': 'Valid operation'}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_month_range(month, year):
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(days=1)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def count_days_between(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end   = datetime.strptime(end_date,   '%Y-%m-%d')
    return (end - start).days + 1


def get_unique_viz_filename(operation, index=None):
    import time
    import uuid
    ts = time.strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    if index is not None:
        return f"analysis_{operation}_{ts}_{index}_{unique_id}.png"
    return f"analysis_{operation}_{ts}_{unique_id}.png"

# =============================================================================
# OLLAMA-BASED QUERY SPLITTER FOR "BOTH" INTENT
# =============================================================================

_SPLIT_PROMPT = """You are a query splitter for a soil moisture research system.

The user has asked a combined question. Split it into two standalone sub-questions.

DEFINITIONS:
  LITERATURE sub-question:
    - Asks about information FROM a scientific paper, PDF, journal, or research study.
    - Clues: "according to [paper].pdf", "in the paper", "in the study", "the literature says",
      "Figure X", "Table X", "IEEE", "methodology", "algorithm", "LPRM", "AMSR", "SMAP",
      "retrieval", "validation", "backscatter", "dielectric", "predicted vs observed".
    - Example: "How does the initial soil moisture pulse affect VWC according to paper.pdf?"

  DATASET sub-question:
    - Asks for soil moisture numbers/statistics from the database for a region and time period.
    - Clues: maximum/minimum/mean/average/trend in [region] during [year or season],
      "what was the", "show me", "wettest season", "driest period", "moisture trend".
    - Example: "What was the maximum soil moisture scalar in India during the wettest season of 2020?"

Rules:
- Preserve the user's exact wording as much as possible.
- If a part is absent, return "" for it.
- Do NOT invent information not in the original query.
- Output ONLY valid JSON — nothing else.

Example:
  Query: "How does the LPRM algorithm work according to the paper, and what was the mean soil moisture in Rajasthan in June 2020?"
  Output:
  {{
    "literature": "How does the LPRM algorithm work according to the paper?",
    "dataset":    "What was the mean soil moisture in Rajasthan in June 2020?"
  }}

Now split this query:

User query: {query}

JSON:"""


def split_both_query_with_llm(
    query: str,
    ollama_url: str   = "http://localhost:11434/api/generate",
    ollama_model: str = "qwen2.5:3b",
    timeout: int      = 60,
) -> tuple:
    """
    Use Ollama to split a dual-intent query into its dataset and literature parts.

    Returns
    -------
    (dataset_query: str, literature_query: str)
      Each is either the extracted sub-question or the original query as fallback.
    """
    import requests, json as _json

    prompt = _SPLIT_PROMPT.format(query=query)
    try:
        resp = requests.post(
            ollama_url,
            json={
                "model" : ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_p"      : 0.1,
                    "num_ctx"    : 2048,
                    "num_predict": 256,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # Extract JSON block from response (model may emit extra text)
        match = re.search(r'\{[^{}]*"dataset"[^{}]*"literature"[^{}]*\}', raw, re.DOTALL)
        if not match:
            match = re.search(r'\{.*?\}', raw, re.DOTALL)

        if match:
            data  = _json.loads(match.group())
            ds_q  = (data.get("dataset",    "") or "").strip()
            lit_q = (data.get("literature", "") or "").strip()
            ds_q  = ds_q  if len(ds_q)  >= 5 else query
            lit_q = lit_q if len(lit_q) >= 5 else query

            # Validate: if LLM reversed the split, detect and swap
            _lit_signals  = ['.pdf', 'according to', 'in the paper', 'in the study',
                              'figure ', 'table ', 'ieee', 'algorithm', 'lprm',
                              'methodology', 'retrieval', 'backscatter', 'amsr', 'smap']
            _data_signals = ['maximum', 'minimum', 'mean', 'average', 'trend',
                              'wettest', 'driest', 'highest', 'lowest', 'scalar']

            ds_has_lit  = any(s in ds_q.lower()  for s in _lit_signals)
            lit_has_data = any(s in lit_q.lower() for s in _data_signals)
            ds_has_data  = any(s in ds_q.lower()  for s in _data_signals)
            lit_has_lit  = any(s in lit_q.lower() for s in _lit_signals)

            # If dataset part looks like literature AND lit part looks like data → swap
            if ds_has_lit and not ds_has_data and lit_has_data and not lit_has_lit:
                ds_q, lit_q = lit_q, ds_q
                print("[SPLIT] Detected reversed split — swapped")

            # Smart subtraction fallback: if one part is the full original query
            # (because LLM returned empty for it) but the other was extracted
            # correctly, derive the missing part by removing the extracted text.
            def _subtract_part(original, extracted):
                remaining = original.replace(extracted, '', 1).strip(' ,;')
                remaining = re.sub(r'^[\s,;]*(and|but|also|,)\s*', '', remaining,
                                   flags=re.IGNORECASE).strip(' ,;')
                return remaining if len(remaining) >= 5 else None

            if ds_q == query and lit_q != query:
                derived = _subtract_part(query, lit_q)
                if derived:
                    ds_q = derived
                    print(f"[SPLIT] Derived dataset_q by subtraction: {ds_q[:80]}")
            elif lit_q == query and ds_q != query:
                derived = _subtract_part(query, ds_q)
                if derived:
                    lit_q = derived
                    print(f"[SPLIT] Derived lit_q by subtraction: {lit_q[:80]}")

            print(f"[SPLIT] dataset_q  : {ds_q[:80]}")
            print(f"[SPLIT] lit_q      : {lit_q[:80]}")
            return ds_q, lit_q

    except Exception as e:
        print(f"[SPLIT] Ollama error: {e}")

    # Fallback: return original for both
    return query, query
