import pytest
from ragas.metrics.collections import Faithfulness

from chatbot_app.chatbot import ask_rag, create_chatbot
from conftest import llm_setup
from utils.resuable import load_data


@pytest.fixture
def start_chatbot():
    return create_chatbot()


@pytest.fixture
def faithfulness_metric():
    return Faithfulness(llm=llm_setup())


# -----------------------------
# TEST
# -----------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("test_data", load_data())
async def test_faithfulness(test_data, start_chatbot, faithfulness_metric):

    question = test_data["question"]

    retriever, llm, prompt = start_chatbot

    answer, context = ask_rag(
        question,
        retriever,
        llm,
        prompt
    )

    score = await faithfulness_metric.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=context
    )

    assert score >= 0.7, f"""
Faithfulness failed
Score: {score}
Question: {question}
Answer: {answer}
Context: {context}
"""