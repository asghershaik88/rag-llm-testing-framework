import pytest
from ragas.metrics.collections import ContextPrecision
from chatbot_app.chatbot import ask_rag, create_chatbot
from conftest import llm_setup
from utils.resuable import load_data


@pytest.fixture
def start_chatbot():
    retriever, llm, prompt=create_chatbot()
    return retriever, llm, prompt

@pytest.fixture
def context_precision_metric():
    return ContextPrecision(llm=llm_setup())

@pytest.mark.parametrize("data_set",load_data())
@pytest.mark.asyncio
async def test_context_precision(data_set,start_chatbot,context_precision_metric):
    question = data_set["question"]
    reference = data_set["reference_answer"]
    retriever, llm, prompt = start_chatbot
    answer,context = ask_rag(question, retriever, llm, prompt)
    score = await context_precision_metric.ascore(
        user_input=question,
        reference=reference,
        retrieved_contexts=context
    )
    print(context)
    print(score)
    print(answer)
    assert score >= 0.7








