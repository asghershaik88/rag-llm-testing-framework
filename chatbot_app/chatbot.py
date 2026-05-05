import os
import sys
from pathlib import Path

# Add project root to sys.path for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langsmith import traceable
from Monitoring.logger import log_interaction
from utils.Reranker import rerank_documents

from config.settings import settings


# -----------------------------
# CREATE CHATBOT COMPONENTS
# -----------------------------
def create_chatbot():

    BASE_DIR = Path(__file__).resolve().parent.parent
    file_path = BASE_DIR / "data" / "hrdocs"

    documents = []

    for file in file_path.glob("*.txt"):
        loader = TextLoader(str(file), encoding="utf-8")
        documents.extend(loader.load())

    # Split documents
    splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embeddings (using settings)
    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY
    )

    # Vector DB
    vector_db = FAISS.from_documents(docs, embeddings)

    # Retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.OPENAI_API_KEY
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
You are a secure AI assistant.

Use ONLY the context below to answer the question.
If context is insufficient, say it is not fully available and answer cautiously.

Context:
{context}

Question:
{question}
""")

    return retriever, llm, prompt


# -----------------------------
# RAG PIPELINE
# -----------------------------
@traceable
def ask_rag(question, retriever, llm, prompt, history=None):

    # Retrieve
    docs = retriever.invoke(question)

    # Rerank
    docs = rerank_documents(question, docs, top_k=3)

    context = "\n".join([doc.page_content for doc in docs])

    # History (optional)
    history_text = ""
    if history:
        history_text = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in history
        )

    # Final prompt
    final_prompt = prompt.format(
        context=context + "\n\n" + history_text,
        question=question
    )

    # LLM call
    response = llm.invoke(final_prompt)

    # Logging
    log_interaction(
        query=question,
        response=response.content,
        context=[context]
    )

    return response.content, docs


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    retriever, llm, prompt = create_chatbot()

    print("\nRAG Chatbot with Reranker is running...\n")

    while True:
        question = input("Ask something: ")

        if question.lower() == "exit":
            break

        answer, docs = ask_rag(question, retriever, llm, prompt)

        print("\nAnswer:\n", answer)

        print("\n--- Retrieved Context ---")
        for d in docs:
            print("-", d.page_content[:120])
