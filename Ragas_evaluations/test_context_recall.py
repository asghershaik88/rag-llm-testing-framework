import pytest
from ragas.metrics.collections import ContextRecall
from chatbot_app.chatbot import create_chatbot, ask_rag
from conftest import llm_setup
from utils.resuable import load_data

@pytest.fixture
def start_chatbot():
    retriever, llm, prompt = create_chatbot()
    return retriever, llm, prompt

@pytest.fixture
def context_recall():
    return ContextRecall(llm=llm_setup())

@pytest.mark.parametrize("test_data",load_data())
@pytest.mark.asyncio
async def test_context_recall(test_data,start_chatbot,context_recall):
    question=test_data["question"]
    reference=test_data["reference_answer"]
    retriever, llm, prompt=start_chatbot
    answer, context = ask_rag(question, retriever, llm, prompt, history=None)
    recall=ContextRecall(llm=llm_setup())

    score= await recall.ascore(
        user_input=question,
        retrieved_contexts=context,
        reference=reference
    )
    print(context)
    print(score)
    print(answer)
    assert score>=0.8



