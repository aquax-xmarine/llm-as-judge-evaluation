import json
from judge import JUDGE_MODEL

with open(f"evaluation_report_{JUDGE_MODEL}.json") as f:
    evaluation_report = json.load(f)

# Find problematic responses
print("=== LOW FAITHFULNESS (score < 4) ===")
for r in evaluation_report:
    if r["faithfulness_score"] < 4:
        print(f"\nQ: {r['question']}")
        print(f"Score: {r['faithfulness_score']}")
        print(f"Reason: {r['faithfulness_reason']}")
        print(f"Answer preview: {r['answer'][:150]}...")

print("\n=== LOW RELEVANCE (score < 4) ===")
for r in evaluation_report:
    if r["relevance_score"] < 4:
        print(f"\nQ: {r['question']}")
        print(f"Score: {r['relevance_score']}")
        print(f"Reason: {r['relevance_reason']}")