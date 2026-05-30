from chatbot_app.Agentic_RAG import route_decision


def test_route_to_generate():
    state={
        "decision":"NO",
        "iteration":1
    }
    result=route_decision(state)

    assert result == "retrieve"
