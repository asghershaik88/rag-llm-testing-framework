from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from fastapi.responses import JSONResponse
from config.settings import settings

app = FastAPI()

# -----------------------------
# BUILD RAG ON STARTUP
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

    prompt = ChatPromptTemplate.from_template("""
You are a secure HR assistant.

Context:
{context}

Question:
{question}

Answer clearly using only the context.
""")

    return retriever, prompt


retriever, prompt = create_chatbot()


# -----------------------------
# REQUEST MODEL
# -----------------------------
class ChatRequest(BaseModel):
    question: str
    model: str = "gpt-4o-mini"


# -----------------------------
# SAFE LLM CALL
# -----------------------------
def ask_rag(question: str, model: str):

    docs = retriever.invoke(question)
    context = [d.page_content for d in docs]

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=settings.OPENAI_API_KEY
    )

    final_prompt = prompt.format(
        context="\n".join(context),
        question=question
    )

    response = llm.invoke(final_prompt)

    answer = response.content if response and response.content else "No answer generated"

    # IMPORTANT: normalize output
    return {
        "answer": str(answer).strip(),
        "context": context
    }


# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    result = ask_rag(request.question, request.model)

    # HARD GUARANTEE CLEAN OUTPUT
    return JSONResponse(
        content={
            "answer": str(result["answer"]),
            "context": result["context"]
        }
    )


@app.get("/")
def home():
    return {"status": "Chatbot API running"}