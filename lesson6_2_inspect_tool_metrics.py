
import inspect
from ragas.metrics import collections

for metric_name in ["ToolCallAccuracy", "ToolCallF1"]:
    metric_class = getattr(collections, metric_name, None)

    print(f"\n{metric_name}")
    print("-" * 40)

    if metric_class is None:
        print("Not available in this RAGAS version")
        continue

    print("Constructor:", inspect.signature(metric_class))

    if hasattr(metric_class, "ascore"):
        print("ascore:", inspect.signature(metric_class.ascore))

    # poetry run python lesson6_2_inspect_tool_metrics.py
