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

from config.settings import settings


# Create RAGAS-compatible chatbot
class RAGASChatbot:
    def __init__(self):
        self.retriever, self.llm, self.prompt = self._setup()

    def _setup(self):
        BASE_DIR = Path(__file__).resolve().parent.parent
        file_path = BASE_DIR / "data" / "hrdocs"

        documents = []
        for file in file_path.glob("*.txt"):
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

    def generate(self, question: str):
        """Generate answer for RAGAS evaluation"""
        docs = self.retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])

        final_prompt = self.prompt.format(
            context=context,
            question=question
        )

        response = self.llm.invoke(final_prompt)
        return response.content

    def get_retrieval_context(self, question: str):
        """Get retrieval context for RAGAS evaluation"""
        docs = self.retriever.invoke(question)
        return [doc.page_content for doc in docs]


if __name__ == "__main__":
    chatbot = RAGASChatbot()

    print("\nRASAG-compatible Chatbot is running...\n")

    while True:
        question = input("Ask something: ")

        if question.lower() == "exit":
            break

        answer = chatbot.generate(question)
        print("\nAnswer:\n", answer)

        context = chatbot.get_retrieval_context(question)
        print("\n--- Retrieved Context ---")
        for i, ctx in enumerate(context, 1):
            print(f"{i}. {ctx[:150]}")

