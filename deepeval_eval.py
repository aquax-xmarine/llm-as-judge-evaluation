import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

load_dotenv()

class DeepSeekJudge(DeepEvalBaseLLM):
    """Lets DeepEval use DeepSeek as its judge model."""

    def __init__(self, model_name="deepseek-reasoner"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

judge = DeepSeekJudge()

# Configure metrics
faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True)
relevance_metric = AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True)

# Load the saved RAG answers
with open("rag_results.json") as f:
    eval_results = json.load(f)

test_cases = [
    LLMTestCase(
        input=r["question"],
        actual_output=r["answer"],
        retrieval_context=[r["context"]],
    )
    for r in eval_results
]

evaluate(test_cases=test_cases, metrics=[faithfulness_metric, relevance_metric])