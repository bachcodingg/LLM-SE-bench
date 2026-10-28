import json

with open("analysis/statistical_summary.json", encoding="utf-8") as f:
    raw = json.load(f)

desc = raw.get("descriptive", [])
lat = [d for d in desc if d.get("metric_name") == "latency_ms"]
print(f"latency_ms summaries: {len(lat)}")
for d in lat:
    print(f"  model={d['model_id']:<28} dataset={d.get('dataset','?'):<18} mean={d['mean']:.1f}  n={d['n']}")
