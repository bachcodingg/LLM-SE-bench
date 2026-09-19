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

print(f"Summaries loaded: {len(summaries)}")
print("Metrics present:", sorted(set(s.metric_name for s in summaries)))
print()

engine = DecisionMatrixEngine(summaries)
matrices = engine.build()

hdr = f"{'Rank':<5} {'Model':<28} {'TOTAL':<8} {'correct':<9} {'quality':<9} {'speed_ms':<10} {'cost_usd':<11} {'consist':<8}"
print(hdr)
print("-" * len(hdr))
for m in matrices:
    sc = {s.criterion: s.raw_value for s in m.scores}
    print(
        f"{m.rank:<5} {m.model_id:<28} {m.weighted_total:<8.4f} "
        f"{sc.get('correctness', 0):<9.4f} {sc.get('quality', 0):<9.2f} "
        f"{sc.get('speed', 0):<10.1f} {sc.get('cost', 0):<11.6f} "
        f"{sc.get('consistency', 0):<8.4f}"
    )
