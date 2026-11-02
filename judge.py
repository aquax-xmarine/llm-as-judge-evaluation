import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

JUDGE_MODEL = "deepseek-reasoner"

def call_judge(eval_prompt: str) -> dict:
    """Send a prompt to the judge model and parse its JSON reply."""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def judge_faithfulness(question: str, context: str, answer: str) -> dict:
    """Judge whether the answer is faithful to the retrieved context."""
    eval_prompt = f"""
      You are an impartial judge evaluating whether an AI assistant's
      answer is faithful to the provided context.

      Faithfulness means every claim in the answer can be traced back to
      information in the context. The answer should not contain information
      that isn't supported by or inferable from the context.

      Score on a scale of 1 to 5:
      1 - The answer contains multiple claims not supported by the context
      2 - The answer contains at least one significant unsupported claim
      3 - The answer is mostly faithful but includes minor unsupported details
      4 - The answer is faithful with only trivial extrapolations
      5 - Every claim in the answer is directly supported by the context

      Context: {context}

      Question: {question}

      Answer to evaluate: {answer}

      Respond in this exact JSON format: {{"score": <int 1-5>, "reason": "<one paragraph explanation>"}}
    """
    return call_judge(eval_prompt)


def judge_relevance(question: str, answer: str) -> dict:
    """Judge whether the answer is relevant to the question."""
    eval_prompt = f"""
      You are an impartial judge evaluating whether an AI assistant's answer
      is relevant to the user's question.

      Relevance means the answer directly addresses what the user asked.
      A relevant answer may acknowledge limitations in available information,
      but it should not go off-topic or provide unrelated information.

      Score on a scale of 1 to 5:
      1 - The answer does not address the question at all
      2 - The answer partially addresses the question but misses the main point
      3 - The answer addresses the question but includes significant irrelevant content
      4 - The answer addresses the question well with minor tangents
      5 - The answer directly and completely addresses the question

      Question: {question}

      Answer to evaluate: {answer}

      Respond in this exact JSON format: {{"score": <int 1-5>, "reason": "<one paragraph explanation>"}}
    """
    return call_judge(eval_prompt)