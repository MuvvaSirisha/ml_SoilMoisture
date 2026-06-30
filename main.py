"""
main.py
=======
Main Application - Soil Moisture Analysis Engine.
Advanced NLP-powered analysis with flexible output control.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

import Config
from engine import SM_Engine
from agent import OllamaAgent
from Query_classifier import QueryClassifier
from utils import QueryValidator, DateAnalyzer

from intent_classifier import classify_query_intent
from guardrails import QueryGuard



# ============================================================================
# HEADER
# ============================================================================

def display_header():
    print("\n" + "=" * 75)
    print("🌍 SOIL MOISTURE ANALYSIS ENGINE — ADVANCED NLP + VISION INTERFACE")
    print("=" * 75)
    print("""
📌 SUPPORTED QUERIES

  Mean / Average
    "What is average moisture in Rajasthan for June 2022?"
    "Show mean moisture values of Kerala in 2007 and 2019"     ← multi-year

  Trend
    "Show moisture trend in Punjab during monsoon 2022"

  Minimum / Maximum
    "Find minimum moisture in Kerala for July 2022"

  Comparison (any number of periods)
    "Compare Rajasthan and Gujarat in 2021"
    "Compare India between 2020 and 2023"
    "Compare India in 2018, 2020 and 2022"                     ← 3-way

📅 DATE FORMATS
   June 2022 | 2022-06-15 | monsoon 2022 | annual 2022
   2020 and 2021 | 2018, 2020 and 2022 | between 2020 and 2023

📊 OUTPUT OPTIONS
   scalar / map / both   (default: both — always shows map + stats)
""")
    print("\nType 'exit' to quit, 'help' to display this again.")
    print("=" * 75 + "\n")

# ============================================================================
# SPLIT MULTI-QUESTION INPUT
# ============================================================================

def split_queries(raw: str) -> list:
    parts   = re.split(r"\s*\?\s*", raw)
    queries = [p.strip() for p in parts if len(p.strip()) >= 5]
    return queries if queries else [raw.strip()]

# ============================================================================
# SANITISE RAW INPUT
# ============================================================================

def sanitise_input(raw: str) -> str:
    cleaned = raw.strip().strip("`")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    return cleaned.strip()

# ============================================================================
# HANDLE UNCLEAR QUERIES
# ============================================================================

def handle_unclear_query(classification, classifier):
    if classification.get("query_clarity") == "clear":
        return classification

    print("\n⚠️  Query interpretation is uncertain.")
    print(classifier.describe(classification))

    proceed = input("\nProceed with this interpretation? (yes / no): ").strip().lower()
    if proceed not in ["yes", "y"]:
        print("\nPlease enter a clearer query.")
        print("Suggestions:")
        print("  • Mention region/state clearly")
        print("  • Mention operation (mean/trend/max/min/compare)")
        print("  • Mention dates clearly")
        print("  • For multi-year: '2020 and 2022' or '2018, 2020 and 2022'")
        print("  • For comparison: mention 'compare' explicitly")
        return None
    return classification

# ============================================================================
# FORMAT ANALYSIS OUTPUT
# ============================================================================

def format_analysis_output(result_message, visualization_created, output_type,
                            viz_filename="latest_analysis.png"):
    output  = "\n" + "=" * 75 + "\n"
    output += result_message

    if visualization_created and output_type in ["map", "both"]:
        output += f"\n✅ Visualization saved to '{viz_filename}'"
    elif output_type in ["map", "both"] and not visualization_created:
        output += "\n⚠️  Visualization generation failed."

    output += "\n" + "=" * 75 + "\n"
    return output

# ============================================================================
# PRINT INTERPRETATION
# ============================================================================

def print_interpretation(cls, classifier):
    print("\n✅ Interpretation")
    print(classifier.describe(cls))

# ============================================================================
# BUILD COMPARISON INFO
# ============================================================================

def build_comparison_info(cls: dict) -> dict:
    return {
        "comparison_type"   : cls.get("comparison_type",    "time"),
        "comparison_metric" : cls.get("comparison_metric",  "mean"),
        "comparison_periods": cls.get("comparison_periods", []),
        "comparison_period1": cls.get("comparison_period1"),
        "comparison_period2": cls.get("comparison_period2"),
        "comparison_region2": cls.get("comparison_region2"),
    }

# ============================================================================
# DATASET BOUNDS
# ============================================================================

def get_dataset_bounds(engine):
    try:
        import pandas as pd
        times = engine.ds["time"].values
        min_t = pd.Timestamp(times.min()).strftime("%Y-%m-%d")
        max_t = pd.Timestamp(times.max()).strftime("%Y-%m-%d")
        return min_t, max_t
    except Exception:
        return None, None

# ============================================================================
# DATE RANGE VALIDATION
# ============================================================================

def check_date_bounds(start_date, end_date, ds_start, ds_end):
    if ds_start is None:
        return True, ""
    if end_date < ds_start or start_date > ds_end:
        return False, (
            f"❌ Requested period ({start_date} to {end_date}) "
            f"is outside dataset range.\n"
            f"Available dataset: {ds_start} → {ds_end}"
        )
    warn = ""
    if start_date < ds_start:
        warn += f"⚠️  Start date before dataset start ({ds_start}).\n"
    if end_date > ds_end:
        warn += f"⚠️  End date after dataset end ({ds_end}).\n"
    return True, warn


def _apply_date_correction(validator, start_date, end_date):
    dv = validator.validate_dates(start_date, end_date)
    if not dv['valid']:
        return False, start_date, end_date, dv['message']
    return True, dv['start_date'], dv['end_date'], dv['message']

# ============================================================================
# REGION RESOLUTION
# ============================================================================

def resolve_region(cls: dict, engine) -> bool:
    from difflib import get_close_matches

    all_valid = list(engine.available_regions) + ['india', 'north', 'south', 'east', 'west', 'central', 'northeast']

    if not cls.get('region_missing'):
        region = cls.get('region', '')
        if not region:
            cls['region_missing'] = True
        else:
            region_lower = region.lower()
            if region_lower not in [r.lower() for r in all_valid]:
                print(f"\n❌ Data unavailable for region '{region.title()}'.")
                print(f"   This region is not covered by the dataset.")
                _print_available_regions(engine)
                return False
            return True

    print("\n" + "─" * 60)
    print("⚠️  No region was detected in your query.")
    print("   Please specify an Indian state or 'India' for national-level analysis.")
    _print_available_regions(engine)
    print("─" * 60)

    user_input = input("\n📍 Enter region name: ").strip()

    if not user_input:
        print("❌ No region entered. Query cancelled.")
        return False

    user_lower = user_input.lower()

    if user_lower in [r.lower() for r in all_valid]:
        for r in all_valid:
            if r.lower() == user_lower:
                cls['region']         = r
                cls['region_missing'] = False
                print(f"   ✅ Region set to: {r.title()}")
                return True

    close = get_close_matches(user_lower, [r.lower() for r in all_valid],
                               n=1, cutoff=0.6)
    if close:
        matched_lower = close[0]
        canonical = next(r for r in all_valid if r.lower() == matched_lower)
        if matched_lower != user_lower:
            confirm = input(
                f"   Did you mean '{canonical.title()}'? (yes / no): "
            ).strip().lower()
            if confirm not in ('yes', 'y'):
                print("❌ Region not confirmed. Query cancelled.")
                return False
        cls['region']         = canonical
        cls['region_missing'] = False
        print(f"   ✅ Region set to: {canonical.title()}")
        return True

    print(f"\n❌ Data unavailable for region '{user_input.title()}'.")
    print(f"   '{user_input}' is not covered by this dataset.")
    _print_available_regions(engine)
    return False


def _print_available_regions(engine):
    sorted_regions = sorted(engine.available_regions)
    lines = []
    row   = []
    for i, r in enumerate(sorted_regions):
        row.append(r.title())
        if len(row) == 4:
            lines.append("   " + " | ".join(f"{s:<22}" for s in row))
            row = []
    if row:
        lines.append("   " + " | ".join(f"{s:<22}" for s in row))

    print("\n   📋 Regions available in this dataset:")
    for line in lines:
        print(line)
    print()

# ============================================================================
# PROCESS SINGLE QUERY (dataset path)
# ============================================================================

def process_single_query(
    query,
    engine,
    classifier,
    agent,
    validator,
    ds_start,
    ds_end,
    query_index=None,
    intent="dataset",
):
    if query_index is not None:
        print(f"\n{'─'*75}")
        print(f"🔹 QUERY {query_index}: {query}")
        print(f"{'─'*75}")

    print("\n⏳ Processing query...")

    cls = classifier.classify(query)

    if cls.get("query_clarity") in ["unclear", "ambiguous"]:
        cls = agent.process_query(cls)

    if cls.get("query_clarity") != "clear":
        cls = handle_unclear_query(cls, classifier)
        if cls is None:
            return

    print_interpretation(cls, classifier)

    if not resolve_region(cls, engine):
        return

    ov = validator.validate_operation(cls["operation"])
    if not ov["valid"]:
        print(f"❌ {ov['message']}")
        return

    if cls["operation"] == "comparison":
        _run_comparison_query(
            cls, engine, validator, ds_start, ds_end, intent
        )
        return

    all_ranges = cls.get("all_date_ranges", [])

    if len(all_ranges) <= 1:
        _run_single_date_range(
            cls, engine, validator, ds_start, ds_end, intent,
            start_date=cls["start_date"],
            end_date=cls["end_date"],
        )
        return

    print(f"\n📋 {len(all_ranges)} date ranges detected — running each separately.\n")
    for i, (s, e) in enumerate(all_ranges, 1):
        valid, s, e, msg = _apply_date_correction(validator, s, e)
        if not valid:
            print(f"\n[Range {i}] ❌ {msg}")
            continue

        ok, bounds_msg = check_date_bounds(s, e, ds_start, ds_end)
        if not ok:
            print(f"\n[Range {i}] {bounds_msg}")
            continue
        if bounds_msg:
            print(bounds_msg)

        print(f"\n{'─'*60}")
        print(f"📅 Range {i} of {len(all_ranges)}: {s} → {e}")
        print(f"{'─'*60}")

        viz_filename = None
        if cls["output_type"] in ["map", "both"]:
            from utils import get_unique_viz_filename
            viz_filename = get_unique_viz_filename(cls["operation"], index=i)

        _run_single_date_range(
            cls, engine, validator, ds_start, ds_end, intent,
            start_date=s,
            end_date=e,
            viz_filename=viz_filename,
        )


def _run_single_date_range(cls, engine, validator, ds_start, ds_end,
                            intent,
                            start_date=None, end_date=None,
                            viz_filename="latest_analysis.png"):
    s = start_date or cls["start_date"]
    e = end_date   or cls["end_date"]

    if not s or not e:
        print("❌ Could not determine dates from query.")
        return

    valid, s, e, msg = _apply_date_correction(validator, s, e)
    if not valid:
        print(f"❌ {msg}")
        return

    ok, bounds_msg = check_date_bounds(s, e, ds_start, ds_end)
    if not ok:
        print(bounds_msg)
        return
    if bounds_msg:
        print(bounds_msg)

    print("\n🔍 Executing analysis...")
    result_msg, viz_created = engine.execute_analysis(
        region      = cls["region"],
        start_date  = s,
        end_date    = e,
        operation   = cls["operation"],
        output_type = cls["output_type"],
    )

    actual_viz_filename = None
    if viz_created:
        import shutil
        from utils import get_unique_viz_filename
        actual_viz_filename = viz_filename or get_unique_viz_filename(cls["operation"])
        try:
            shutil.copy("latest_analysis.png", actual_viz_filename)
        except Exception as mv_err:
            print(f"⚠️  Could not copy output file: {mv_err}")
            actual_viz_filename = "latest_analysis.png"

    print(format_analysis_output(result_msg, viz_created, cls["output_type"],
                                  viz_filename=actual_viz_filename))

    if hasattr(engine, "ds") and "time" in engine.ds.dims:
        try:
            available_times = engine.ds["time"].values
            missing_info    = DateAnalyzer.find_missing_dates(available_times, s, e)
            if missing_info["has_gaps"]:
                print(DateAnalyzer.format_missing_report(
                    missing_info, cls["region"], s, e
                ))
        except Exception:
            pass


def _run_comparison_query(cls, engine, validator, ds_start, ds_end, intent):
    comparison_info = build_comparison_info(cls)
    ctype = comparison_info["comparison_type"]

    output_type = cls.get("output_type", "both")
    if output_type not in ("scalar", "map", "both"):
        output_type = "both"
    if output_type == "scalar":
        print("ℹ️  Comparison: scalar-only mode (no map will be generated).")
    cls["output_type"] = output_type

    if ctype == "time":
        periods = comparison_info.get("comparison_periods", [])
        if len(periods) < 2:
            print("❌ Two or more time periods required.")
            return

        corrected_periods = []
        for i, (s, e) in enumerate(periods, 1):
            valid, s, e, msg = _apply_date_correction(validator, s, e)
            if not valid:
                print(f"❌ Period {i}: {msg}")
                return
            ok, bounds_msg = check_date_bounds(s, e, ds_start, ds_end)
            if not ok:
                print(f"❌ Period {i}: {bounds_msg}")
                return
            if bounds_msg:
                print(bounds_msg)
            corrected_periods.append((s, e))

        comparison_info["comparison_periods"] = corrected_periods
        comparison_info["comparison_period1"] = corrected_periods[0]
        comparison_info["comparison_period2"] = corrected_periods[1]
        cls["start_date"] = corrected_periods[0][0]
        cls["end_date"]   = corrected_periods[-1][1]

        print(f"\n📋 Time comparison: {len(corrected_periods)} periods detected.")
        for i, (s, e) in enumerate(corrected_periods, 1):
            print(f"   Period {i}: {s} → {e}")

    elif ctype == "region":
        if not comparison_info["comparison_region2"]:
            print("❌ Two regions required.")
            return
        s = cls["start_date"]
        e = cls["end_date"]
        if not s or not e:
            print("❌ Could not determine dates from query.")
            return

        valid, s, e, msg = _apply_date_correction(validator, s, e)
        if not valid:
            print(f"❌ {msg}")
            return
        cls["start_date"] = s
        cls["end_date"]   = e

        ok, bounds_msg = check_date_bounds(s, e, ds_start, ds_end)
        if not ok:
            print(bounds_msg)
            return
        if bounds_msg:
            print(bounds_msg)

        r1 = cls["region"].title()
        r2 = comparison_info["comparison_region2"].title()
        print(f"\n📋 Region comparison: {r1} vs {r2}  ({s} → {e})")

    print("\n🔍 Executing analysis...")
    result_msg, viz_created = engine.execute_analysis(
        region          = cls["region"],
        start_date      = cls["start_date"],
        end_date        = cls["end_date"],
        operation       = cls["operation"],
        output_type     = cls["output_type"],
        comparison_info = comparison_info,
    )

    actual_viz_filename = None
    if viz_created:
        import shutil
        from utils import get_unique_viz_filename
        actual_viz_filename = get_unique_viz_filename(cls["operation"])
        try:
            shutil.copy("latest_analysis.png", actual_viz_filename)
        except Exception as e:
            print(f"⚠️  Could not copy output file: {e}")
            actual_viz_filename = "latest_analysis.png"

    print(format_analysis_output(result_msg, viz_created, cls["output_type"],
                                  viz_filename=actual_viz_filename))

    if hasattr(engine, "ds") and "time" in engine.ds.dims:
        try:
            available_times = engine.ds["time"].values
            missing_info    = DateAnalyzer.find_missing_dates(
                available_times, cls["start_date"], cls["end_date"]
            )
            if missing_info["has_gaps"]:
                print(DateAnalyzer.format_missing_report(
                    missing_info, cls["region"],
                    cls["start_date"], cls["end_date"],
                ))
        except Exception:
            pass

# ============================================================================
# MAIN APPLICATION LOOP
# ============================================================================

def run_app():
    display_header()

    try:
        print("🔌 Initialising Soil Moisture Engine...")
        engine = SM_Engine()
        print("✅ Engine initialised!")
    except Exception as e:
        print(f"❌ Engine initialisation failed: {e}")
        print("Check Config paths.")
        sys.exit(1)

    classifier = QueryClassifier()
    agent      = OllamaAgent(model_name=Config.OLLAMA_MODEL)
    validator  = QueryValidator()

    ds_start, ds_end = get_dataset_bounds(engine)
    if ds_start and ds_end:
        print(f"\n📅 Dataset covers: {ds_start} → {ds_end}\n")
    else:
        print("\n⚠️  Could not determine dataset range.\n")

    print("🤖 System ready!\n")

    while True:
        try:
            raw_input_text = input("\n📝 Enter query (or 'exit' / 'help'): ").strip()

            if raw_input_text.lower() in ["exit", "quit", "bye", "q"]:
                print("\n👋 Thank you for using the system!")
                break

            if raw_input_text.lower() in ["help", "?"]:
                display_header()
                continue

            if not raw_input_text or len(raw_input_text) < 5:
                print("⚠️  Please enter a more detailed query.")
                continue

            raw_input_text = sanitise_input(raw_input_text)

            guard = QueryGuard()
            safety_check = guard.is_safe(raw_input_text)
            if not safety_check["safe"]:
                print(f"\n🛡️ Guardrail Alert: {safety_check['reason']}")
                continue

            sub_queries = split_queries(raw_input_text)
            if len(sub_queries) > 1:
                print(f"\n📋 Detected {len(sub_queries)} queries.")

            for idx, sq in enumerate(sub_queries, start=1):
                try:
                    print("\n🧭 Classifying query intent...")
                    intent = classify_query_intent(
                        query        = sq,
                        ollama_url   = Config.OLLAMA_URL,
                        ollama_model = Config.OLLAMA_MODEL,
                        timeout      = Config.OLLAMA_TIMEOUT,
                    )

                    if intent in ("dataset", "both", "literature"):
                        process_single_query(
                            sq,
                            engine,
                            classifier,
                            agent,
                            validator,
                            ds_start,
                            ds_end,
                            query_index = (idx if len(sub_queries) > 1 else None),
                            intent      = intent,
                        )

                except Exception as e:
                    print(f"\n❌ Error processing query: {e}")
                    if "--debug" in sys.argv:
                        import traceback
                        traceback.print_exc()

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            if "--debug" in sys.argv:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    run_app()