from pathlib import Path
from typing import TypedDict, List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END

from langsmith import traceable

from Monitoring.logger import log_interaction
from config.settings import settings


# =========================================================
# AGENT STATE
# =========================================================

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    decision: str
    iteration: int


# =========================================================
# BUILD RETRIEVER
# =========================================================

def build_retriever():

    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "hrdocs"

    documents = []

    for file in data_path.glob("*.txt"):
        loader = TextLoader(str(file), encoding="utf-8")
        documents.extend(loader.load())

    splitter = CharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY
    )

    vector_db = FAISS.from_documents(docs, embeddings)

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 5}
    )

    return retriever


# =========================================================
# CREATE LLM
# =========================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=settings.OPENAI_API_KEY
)

retriever = build_retriever()


# =========================================================
# PROMPTS
# =========================================================

judge_prompt = ChatPromptTemplate.from_template("""
You are a strict evaluator.

Question:
{question}

Context:
{context}

Task:
Determine whether the context is sufficient to answer the question accurately.

Respond ONLY with:
YES
or
NO
""")


answer_prompt = ChatPromptTemplate.from_template("""
You are a secure AI assistant.

Use ONLY the provided context to answer the question.

If the context is insufficient, clearly say:
"The available context is insufficient to fully answer this question."

Context:
{context}

Question:
{question}
""")


# =========================================================
# NODE 1 — RETRIEVE CONTEXT
# =========================================================

@traceable
def retrieve_node(state: AgentState):

    print("\n[Agent] Retrieving context...")

    docs = retriever.invoke(state["question"])

    context = [doc.page_content for doc in docs]

    return {
        "context": context,
        "iteration": state.get("iteration", 0) + 1
    }


# =========================================================
# NODE 2 — EVALUATE CONTEXT
# =========================================================

@traceable
def evaluate_context_node(state: AgentState):

    print("\n[Agent] Evaluating context quality...")

    context_text = "\n".join(state["context"])

    response = llm.invoke(
        judge_prompt.format(
            question=state["question"],
            context=context_text
        )
    )

    decision = response.content.strip().upper()

    print(f"[Agent Decision] {decision}")

    return {
        "decision": decision
    }


# =========================================================
# NODE 3 — GENERATE ANSWER
# =========================================================

@traceable
def generate_answer_node(state: AgentState):

    print("\n[Agent] Generating final answer...")

    context_text = "\n".join(state["context"])

    response = llm.invoke(
        answer_prompt.format(
            context=context_text,
            question=state["question"]
        )
    )

    log_interaction(
        query=state["question"],
        response=response.content,
        context=state["context"]
    )

    return {
        "answer": response.content
    }


# =========================================================
# CONDITIONAL ROUTING
# =========================================================

def route_decision(state: AgentState):

    decision = state.get("decision", "NO")
    iteration = state.get("iteration", 0)

    # Context is good → generate answer
    if decision == "YES":
        return "generate"

    # Stop infinite loops after retries
    if iteration >= 2:
        print("\n[Agent] Max retries reached. Generating best possible answer.")
        return "generate"

    print("\n[Agent] Context insufficient. Retrying retrieval...")
    return "retrieve"


# =========================================================
# BUILD LANGGRAPH WORKFLOW
# =========================================================

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)

workflow.add_node(
    "evaluate",
    evaluate_context_node
)

workflow.add_node(
    "generate",
    generate_answer_node
)

# Entry point
workflow.set_entry_point("retrieve")

# Flow
workflow.add_edge("retrieve", "evaluate")

workflow.add_conditional_edges(
    "evaluate",
    route_decision,
    {
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

workflow.add_edge("generate", END)

# Compile app
app = workflow.compile()


# =========================================================
# RUN AGENT
# =========================================================

@traceable
def ask_agent(question: str):

    result = app.invoke({
        "question": question,
        "context": [],
        "answer": "",
        "decision": "",
        "iteration": 0
    })

    return result["answer"], result["context"]


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("\n===================================")
    print(" Agentic RAG System is Running ")
    print(" LangGraph + FAISS + OpenAI ")
    print("===================================\n")

    while True:

        user_input = input("Ask something: ").strip()

        if user_input.lower() == "exit":
            break

        answer, contexts = ask_agent(user_input)

        print("\n================ ANSWER ================\n")
        print(answer)

        print("\n========== RETRIEVED CONTEXT ===========\n")

        for idx, c in enumerate(contexts, start=1):
            print(f"{idx}. {c[:200]}")
            print()