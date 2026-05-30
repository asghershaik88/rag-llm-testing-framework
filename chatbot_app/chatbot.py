from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from Monitoring.logger import log_interaction
from config.settings import settings


# -----------------------------
# CREATE CHATBOT COMPONENTS
# -----------------------------
def create_chatbot():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "hrdocs"

    documents = []

    for file in data_path.glob("*.txt"):
        loader = TextLoader(str(file), encoding="utf-8")
        documents.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    vector_db = FAISS.from_documents(docs, embeddings)

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.OPENAI_API_KEY
    )

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

    docs = retriever.invoke(question)

    context = [doc.page_content for doc in docs]
    context_text = "\n".join(context)

    history_text = ""
    if history:
        history_text = "\n".join(
            f"{h['role']}: {h['content']}" for h in history
        )

    final_prompt = prompt.format(
        context=context_text + "\n\n" + history_text,
        question=question
    )

    response = llm.invoke(final_prompt)

    log_interaction(
        query=question,
        response=response.content,
        context=context
    )

    return response.content, context


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    retriever, llm, prompt = create_chatbot()

    print("\nRAG Chatbot is running...\n")

    while True:
        user_input = input("Ask something: ").strip()

        if user_input.lower() == "exit":
            break

        answer, contexts = ask_rag(user_input, retriever, llm, prompt)

        print("\nAnswer:\n", answer)

        print("\n--- Retrieved Context ---")
        for c in contexts:
            print("-", c[:120])