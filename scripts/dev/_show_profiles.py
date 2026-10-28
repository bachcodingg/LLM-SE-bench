import json
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from contracts import StatisticalSummary
from framework.decision_matrix import DecisionMatrixEngine

with open("analysis/statistical_summary.json", encoding="utf-8") as f:
    raw = json.load(f)

desc = raw.get("descriptive", [])
summaries = []
for d in desc:
    d2 = {k: v for k, v in d.items()
          if k not in ("skewness", "kurtosis", "quartiles", "computed_at")}
    d2.setdefault("min_val", d.get("min", 0.0))
    d2.setdefault("max_val", d.get("max", 0.0))
    d2.setdefault("ci_lower_95", 0.0)
    d2.setdefault("ci_upper_95", 0.0)
    d2.pop("min", None)
    d2.pop("max", None)
    try:
        summaries.append(StatisticalSummary(**d2))
    except Exception:
        pass

PROFILES = ["devops", "audit", "budget"]
for prof in PROFILES:
    try:
        engine = DecisionMatrixEngine(summaries, profile_name=prof)
        matrices = engine.build()
        w = engine.weights
        print(f"\n{'='*60}")
        print(f"Profile: {prof.upper()}")
        wstr = " | ".join(f"{k}={v:.0%}" for k, v in w.items())
        print(f"Weights: {wstr}")
        print(f"{'='*60}")
        hdr = f"  {'Rank':<5} {'Model':<28} {'Score':<8} {'correct':<9} {'speed_ms':<10}"
        print(hdr)
        print("-" * len(hdr))
        for m in matrices:
            sc = {s.criterion: s.raw_value for s in m.scores}
            speed = sc.get('speed', 0)
            speed_str = f"{speed:.0f}" if speed > 0 else "n/a"
            print(f"  {m.rank:<5} {m.model_id:<28} {m.weighted_total:<8.4f} "
                  f"{sc.get('correctness', 0):<9.4f} {speed_str:<10}")
    except Exception as e:
        print(f"Profile {prof}: ERROR — {e}")
