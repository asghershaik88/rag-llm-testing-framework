import pytest
from deepeval.metrics import  HallucinationMetric
from deepeval.test_case import LLMTestCase
from chatbot_app.chatbot import create_chatbot, ask_rag
from utils.resuable import load_data


@pytest.fixture
def start_chatbot():
    retriever, llm, prompt = create_chatbot()
    return retriever, llm, prompt


@pytest.mark.parametrize("test_data", load_data())
@pytest.mark.asyncio
def test_hallucination(test_data, start_chatbot):

    retriever, llm, prompt = start_chatbot
    question = test_data["question"]

    answer, context = ask_rag(question, retriever, llm, prompt)

    sample = LLMTestCase(
        input=question,
        actual_output=answer,
        context=context
    )

    metric = HallucinationMetric()
    score = metric.measure(sample)
    print(question)
    print(answer)
    print(score)
    assert score < 0.2