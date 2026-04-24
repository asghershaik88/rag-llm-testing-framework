import pytest
from deepteam import red_team
from deepteam.attacks.single_turn import Roleplay
from deepteam.test_case import RTTurn
from deepteam.vulnerabilities import PIILeakage

from chatbot_app.chatbot import ask_rag, create_chatbot


@pytest.fixture
def start_chatbot():
     return create_chatbot()

def test_role_play(start_chatbot):
    retriever,llm,prompt=start_chatbot

    async def model_callback(question:str,turns=None)->RTTurn:
        answer,context=ask_rag(question, retriever, llm, prompt)
        return RTTurn(
            role="assistant",
            content=answer,
            retrieval_context=context
        )
    role_play=Roleplay(
        persona="database Manager",
        role="system administrator",
    )
    leakage = PIILeakage(types=["api_and_database_access"])
    red_team(
        attacks=[role_play],
        vulnerabilities=[leakage],
        model_callback=model_callback
    )