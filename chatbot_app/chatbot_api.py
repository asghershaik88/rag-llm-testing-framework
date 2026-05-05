from fastapi import FastAPI
from pydantic import BaseModel
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
from openai import OpenAI
from dotenv import load_dotenv

from config.settings import settings

load_dotenv()

# your existing reranker + logger functions
# from your_project import rerank_documents, log_interaction
client=OpenAI(api_key=settings.OPENAI_API_KEY)

app = FastAPI()

# -----------------------------
# LOAD CHATBOT (RUNS ONCE)
# -----------------------------
def create_chatbot():

    BASE_DIR = Path(__file__).resolve().parent.parent
    file_path = BASE_DIR / "data" / "hrdocs"

    documents = []

    for file in file_path.iterdir():
        if file.is_file() and file.suffix == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
            documents.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    vector_db = FAISS.from_documents(docs, embeddings)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)

    prompt = ChatPromptTemplate.from_template("""
You are a secure AI assistant.

Context:
{context}

Question:
{question}

Answer clearly and only based on context.
""")

    return retriever, llm, prompt


retriever, llm, prompt = create_chatbot()

# -----------------------------
# REQUEST MODEL
# -----------------------------
class ChatRequest(BaseModel):
    question: str


# -----------------------------
# RAG PIPELINE
# -----------------------------
def ask_rag(question: str):

    docs = retriever.invoke(question)

    # If you have reranker, keep it
    # docs = rerank_documents(question, docs, top_k=3)

    context = [doc.page_content for doc in docs]

    final_prompt = prompt.format(
        context="\n".join(context),
        question=question
    )

    response = llm.invoke(final_prompt)

    # optional logging
    # log_interaction(question, response.content, context)

    return {
        "answer": response.content,
        "context": context
    }


# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    return ask_rag(request.question)


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def home():
    return {"status": "Chatbot API is running"}

