import json

from pathlib import Path
import pytest
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from chatbot_app.chatbot import ask_rag, create_chatbot

question="How many paid leaves do employees get?"

@pytest.fixture
def bot_response():
    retriever, llm, prompt=create_chatbot()
    samples=[]
    for i in range(1):
        answer, context = ask_rag(question, retriever, llm, prompt)
        sample = LLMTestCase(
            input=question,
            actual_output=answer,
            context=context
        )
        samples.append(sample)
    return samples



def test_hallucination(bot_response):
    hallucination = HallucinationMetric()
    scores=[]
    for sample in bot_response:
        score = hallucination.measure(sample)
        scores.append(score)

    save_results(scores)

def save_results(scores):

    path=Path(__file__).parent.parent/"Statistical_tests/hallucination_baseline.json"
    with open(path,"w",encoding="utf-8") as f:
        json.dump(scores,f,indent=4)





