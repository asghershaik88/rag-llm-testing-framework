from asyncio import to_thread
import pytest
from deepteam import red_team
from deepteam.attacks.single_turn import SystemOverride
from deepteam.test_case import RTTurn
from deepteam.vulnerabilities import Bias
from chatbot_app.chatbot import create_chatbot, ask_rag


@pytest.fixture
def start_chatbot():
    return create_chatbot()


def test_system_override(start_chatbot):
    retriever, llm, prompt = start_chatbot

    async def model_callback(question: str, turns=None) -> RTTurn:
        answer, context = await to_thread(ask_rag, question, retriever, llm, prompt)

        return RTTurn(
            role="assistant",
            content=answer,
            retrieval_context=context
        )

    system_override = SystemOverride()
    bias = Bias(types=["gender"])

    result = red_team(
        attacks=[system_override],
        vulnerabilities=[bias],
        model_callback=model_callback
    )

