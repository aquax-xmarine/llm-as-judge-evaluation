import json
from rag import rag_query
from data import eval_questions

eval_results = []
for q in eval_questions:
    result = rag_query(q)
    eval_results.append(result)
    print(f"Q: {q}")
    print(f"A: {result['answer'][:150]}...")
    print()

with open("rag_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)

print(f"Saved {len(eval_results)} results to rag_results.json")