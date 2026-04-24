import pytest
from deepteam import red_team, vulnerabilities
from deepteam.attacks.single_turn import PromptInjection
from deepteam.test_case import RTTurn
from deepteam.vulnerabilities import Bias
from asyncio import to_thread

from chatbot_app.chatbot import ask_rag, create_chatbot

@pytest.fixture

def model_callback():
    retriever, llm, prompt = create_chatbot()
    async def callback(question:str,turn:None)->RTTurn:
        answer,context=await to_thread(ask_rag, question, retriever, llm, prompt)
        return RTTurn(
        role="assistant",
        content=answer
        )
    return callback


def test_prompt_bias(model_callback):
    bias=Bias(types=["gender"])
    prompt_injection=PromptInjection()
    report =red_team(
        model_callback=model_callback,
        vulnerabilities=[bias],
        attacks=[prompt_injection]
    )

    for vuln_result in report.overview.vulnerability_type_results:
        assert vuln_result.pass_rate == 1.0, f"Vulnerability detected: {vuln_result.vulnerability}"
