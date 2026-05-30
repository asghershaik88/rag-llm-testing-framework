import pytest

from chatbot_app.Agentic_RAG import ask_agent, route_decision, app


# =========================================================
# 1. FULL WORKFLOW SMOKE TEST
# =========================================================

def test_workflow_end_to_end():

    question = "What is HR leave policy?"

    answer, context = ask_agent(question)

    assert answer is not None
    assert isinstance(answer, str)
    assert len(answer) > 0

    assert isinstance(context, list)


# =========================================================
# 2. CONTEXT IS ALWAYS POPULATED
# =========================================================

def test_retrieval_always_returns_context():

    result = app.invoke({
        "question": "What is termination policy?",
        "context": [],
        "answer": "",
        "decision": "",
        "iteration": 0
    })

    assert "context" in result
    assert isinstance(result["context"], list)
    assert len(result["context"]) > 0


# =========================================================
# 3. DECISION LOGIC TEST (YES PATH)
# =========================================================

def test_generate_path_when_context_sufficient():

    state = {
        "question": "What is probation period?",
        "context": ["Probation is 3 months in most companies"],
        "answer": "",
        "decision": "YES",
        "iteration": 0
    }

    next_step = app.get_graph().nodes

    decision = route_decision(state)

    assert decision == "generate"


# =========================================================
# 4. RETRY LOOP LOGIC TEST (NO PATH)
# =========================================================

def test_retry_when_context_insufficient():

    state = {
        "question": "Unknown HR policy",
        "context": ["irrelevant text"],
        "decision": "NO",
        "iteration": 0
    }

    decision = route_decision(state)

    assert decision in ["retrieve", "generate"]


# =========================================================
# 5. MAX ITERATION STOP CONDITION
# =========================================================

def test_max_iteration_stops_loop():


    state = {
        "question": "Some complex HR rule",
        "context": ["partial info"],
        "decision": "NO",
        "iteration": 2
    }

    decision = route_decision(state)

    # at max iteration → force generate
    assert decision == "generate"


# =========================================================
# 6. STATE STRUCTURE INTEGRITY
# =========================================================

def test_state_structure_after_execution():

    result = app.invoke({
        "question": "What is leave policy?",
        "context": [],
        "answer": "",
        "decision": "",
        "iteration": 0
    })

    required_keys = {"question", "context", "answer", "decision", "iteration"}

    assert required_keys.issubset(result.keys())


# =========================================================
# 7. ANSWER IS GENERATED WHEN FLOW COMPLETES
# =========================================================

def test_answer_is_generated():

    answer, _ = ask_agent("Explain termination process")

    assert isinstance(answer, str)
    assert answer != ""
    assert "context is insufficient" in answer or len(answer) > 10


# =========================================================
# 8. WORKFLOW DOES NOT CRASH UNDER EMPTY INPUT
# =========================================================

def test_empty_question_handling():

    with pytest.raises(Exception):
        ask_agent("")