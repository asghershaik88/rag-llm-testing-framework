import pytest
from unittest.mock import MagicMock, patch

from chatbot_app.Agentic_RAG import AgentState, evaluate_context_node


@patch('chatbot_app.Agentic_RAG')
def test_evaluate_context_node_handles_yes_response(mock_llm):
    fake_llm_response = MagicMock()
    fake_llm_response.content = "  yes  \n"

    mock_llm.invoke.return_value = fake_llm_response

    initial_state: AgentState = {
        "question": "Does the company cover dental?",
        "context": ["The company covers 80% of dental up to $1500."],
        "answer": "",
        "decision": "",
        "iteration": 1
    }

    output_state = evaluate_context_node(initial_state)

    assert "decision" in output_state
    assert output_state["decision"] == "YES"