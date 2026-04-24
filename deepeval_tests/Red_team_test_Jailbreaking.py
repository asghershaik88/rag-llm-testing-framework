from asyncio import to_thread
from typing import List, Optional
from deepteam import red_team
from deepteam.attacks.multi_turn import LinearJailbreaking
from deepteam.test_case import RTTurn
from deepteam.vulnerabilities import Bias
from chatbot_app.chatbot import ask_rag, create_chatbot

retriever, llm, prompt=create_chatbot()
async def model_callback(question:str,turns:Optional[List[RTTurn]] = None)->RTTurn:
    history=[]
    if turns:
        for t in turns:
            history.append({"role": t.role, "content": t.content})

    history.append({"role": "user", "content": question})

    answer,context=await to_thread(ask_rag,question, retriever, llm, prompt,history)
    return RTTurn(
        role="assistant",
        content=answer)


def test_jailbreak():
    linear_jailbreak=LinearJailbreaking(num_turns=3)
    bias=Bias(types=["race"])
    red_team(
        model_callback=model_callback,
        vulnerabilities=[bias],
        attacks=[linear_jailbreak]
    )