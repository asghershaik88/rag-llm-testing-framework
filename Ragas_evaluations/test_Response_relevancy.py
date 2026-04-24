import pytest
from ragas.metrics.collections import AnswerRelevancy
from chatbot_app.chatbot import create_chatbot, ask_rag
from conftest import llm_setup, embeddings
from utils.resuable import load_data

@pytest.fixture
def run_chatbot():
    retriever, llm, prompt=create_chatbot()
    return retriever, llm, prompt

@pytest.fixture
def response_relevancy_metric():
    return AnswerRelevancy(llm=llm_setup(),embeddings=embeddings())

@pytest.mark.parametrize("test_case",load_data())
@pytest.mark.asyncio
async def test_response_relevancy(test_case,run_chatbot,response_relevancy_metric):
    question=test_case["question"]
    retriever, llm, prompt=run_chatbot
    answer,context=ask_rag(question, retriever, llm, prompt)
    score = await response_relevancy_metric.ascore(
        user_input=question,
        response=answer
    )
    assert score >= 0.7