import json
from judge import judge_faithfulness, judge_relevance, JUDGE_MODEL

with open("rag_results.json") as f:
    eval_results = json.load(f)

evaluation_report = []

for result in eval_results:
    # Run both judges
    faithfulness = judge_faithfulness(
        result["question"], result["context"], result["answer"]
    )
    relevance = judge_relevance(result["question"], result["answer"])

    evaluation_report.append({
        "question": result["question"],
        "answer": result["answer"],
        "faithfulness_score": faithfulness["score"],
        "faithfulness_reason": faithfulness["reason"],
        "relevance_score": relevance["score"],
        "relevance_reason": relevance["reason"],
    })

    print(f"Q: {result['question']}")
    print(f"  Faithfulness: {faithfulness['score']}/5 | Relevance: {relevance['score']}/5")
    print(f"  Faith reason: {faithfulness['reason'][:100]}...")
    print()

# Summary statistics
faith_scores = [r["faithfulness_score"] for r in evaluation_report]
rel_scores = [r["relevance_score"] for r in evaluation_report]
print(f"Judge model: {JUDGE_MODEL}")
print(f"Average faithfulness: {sum(faith_scores)/len(faith_scores):.2f}")
print(f"Average relevance: {sum(rel_scores)/len(rel_scores):.2f}")
print(f"Questions with faithfulness < 3: {sum(1 for s in faith_scores if s < 3)}")

# Save the full report
with open(f"evaluation_report_{JUDGE_MODEL}.json", "w") as f:
    json.dump(evaluation_report, f, indent=2)