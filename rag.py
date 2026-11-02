import os
from  dotenv import load_dotenv
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from data import documents

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
# Local SBERT model for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

if not vectorstore.get()["ids"]:
    vectorstore.add_documents(documents)
    print(f"Indexed {len(documents)} documents")

def rag_query(question: str, top_k: int = 2) -> dict:
    """Run a RAG query: retrieve context, generate answer."""

    # Retrieve
    results = vectorstore.similarity_search(question, k=top_k)
    context = "\n\n".join([doc.page_content for doc in results])

    # Generate
    system_prompt = """You are a helpful customer support assistant. Answer the
    customer's question based ONLY on the provided context. If the context doesn't
    contain enough information to answer fully, say so."""

    user_prompt = f"""
      Context: {context}

      Question: {question}

      Answer:
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    return {
        "question": question,
        "context": context,
        "answer": answer,
        "sources": [doc.metadata for doc in results],
    }

result = rag_query("Can I return opened software?")
print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")