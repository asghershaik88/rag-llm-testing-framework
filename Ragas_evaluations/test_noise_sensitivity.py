import pytest

from ragas.metrics.collections import NoiseSensitivity

from chatbot_app.chatbot import create_chatbot, ask_rag
from conftest import llm_setup
from utils.resuable import load_data

@pytest.fixture
def start_chatbot():
    retriever, llm, prompt=create_chatbot()
    return retriever, llm, prompt

@pytest.fixture
def noise_sensitivity_metric():
    return NoiseSensitivity(llm=llm_setup())


@pytest.mark.parametrize("test_data",load_data())
@pytest.mark.asyncio
async def test_noise(test_data,start_chatbot,noise_sensitivity_metric):
    question = test_data["question"]
    reference = test_data["reference_answer"]
    retriever, llm, prompt = start_chatbot
    answer, context = ask_rag(question, retriever, llm, prompt)
    score = await noise_sensitivity_metric.ascore(
        user_input=question,
        response=answer,
        reference=reference,
        retrieved_contexts=context
    )
    assert score <0.3