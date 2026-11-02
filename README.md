# LLM-as-a-Judge RAG Evaluation

Evaluates a small RAG customer-support assistant using an LLM as a judge, and explores where LLM judges fall short.

## Pipeline

```
Questions → Retrieval (Chroma + SBERT) → Answer generation (DeepSeek) → LLM judge → Report
```

Each step saves its output, so later steps can be re-run without repeating earlier ones, and the judge always scores the exact same answers.

## Files

| File | Purpose |
| --- | --- |
| `data.py` | Return-policy knowledge base (5 documents) and 8 evaluation questions |
| `rag.py` | Builds the vector store once, retrieves the top 2 documents, generates answers |
| `generate_answers.py` | Runs RAG on all questions → `rag_results.json` |
| `judge.py` | Custom faithfulness and relevance judges (1–5 rubric, JSON output) |
| `evaluate.py` | Scores saved answers → `evaluation_report_<judge>.json` |
| `analyze.py` | Lists answers scoring below 4 |
| `deepeval_eval.py` | Same evaluation with DeepEval's claim-based metrics |

## Setup

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, run locally
- **Generator:** `deepseek-chat`
- **Judge:** set by `JUDGE_MODEL` in `judge.py` (e.g. `deepseek-reasoner`)

```bash
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your-key" > .env

python3 generate_answers.py
python3 evaluate.py
python3 analyze.py
python3 deepeval_eval.py
```

The evaluation questions cover easy lookups, synthesis across documents, likely hallucination traps, questions the knowledge base can't answer, and tricky edge cases.

## Results: custom judge

| Metric | Average (1–5) |
| --- | --- |
| Faithfulness | ? |
| Relevance | ? |

Every answer scored 4 or higher on both metrics, so `analyze.py` flagged nothing.

## Key finding: faithfulness is not correctness

The high scores looked reassuring, so I read the answers myself.

**A good answer.** For *"Do you offer refunds for international orders?"*, the model correctly said the policy doesn't cover international orders, without inventing anything.

**A hidden failure.** For *"Can I return an item I bought on sale?"*, the answer said **yes**, and that refunds are processed "at the lowest sale price in the last 90 days." But that rule applies to **gifts returned without a receipt**, not to sale items. The policy never mentions sale items at all.

The judge still scored it highly, because every sentence was copied from the retrieved context. Three things went wrong:

1. **Retrieval failed.** The word "sale" matched the gifts document, so the most relevant one, the 30-day refund policy, was pushed out of the top 2.
2. **Facts were applied to the wrong situation.** The generator matched on the keyword "sale," not the meaning.
3. **The faithfulness judge couldn't see either problem.** It only checks whether claims appear in the context, not whether they answer the question correctly, or whether the right documents were retrieved.

## DeepEval comparison

| Metric | Average | Pass rate (threshold 0.7) |
| --- | --- | --- |
| Faithfulness | 1.00 | 100% (8/8) |
| Answer Relevancy | 0.94 | 87.5% (7/8) |

DeepEval got the two most interesting cases backwards:

- **It passed the misleading "sale" answer** with perfect faithfulness. Its claim-by-claim check confirmed every sentence appeared in the context, but couldn't detect that a gift-return rule was applied to sale items.
- **It failed a correct answer.** For "What is the refund window for regular purchases?", the answer gave the 30-day window plus the item conditions. Answer Relevancy scored it 0.50, because it counts the share of statements that are relevant, and treated the useful conditions as irrelevant. The metric penalises helpful completeness.

The run took 224 seconds for 8 test cases, since claim-based metrics make several judge calls per answer.

**Takeaway:** a framework's metrics are only as good as their definitions. Without reading individual answers, these scores would have suggested the opposite of the truth.

## Pitfalls observed

- **Faithfulness ≠ correctness:** an answer can be fully grounded in the context and still be wrong.
- **Faithfulness can't detect retrieval failures:** it only judges against whatever was retrieved, so context recall or reference-based evaluation is needed.
- **Metric definitions can penalise good answers:** DeepEval's relevancy treated helpful conditions as irrelevant.
- **Self-preference bias:** the generator and judge come from the same model family, which may make the judge more lenient. A judge from a different family would be more independent.
- **High averages can hide failures:** reading individual answers revealed problems that every score missed.
- **Cost:** claim-based metrics are slow and make many API calls per answer.

