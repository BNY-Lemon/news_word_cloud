# chatbot.py
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

vector_db = None  # will be set dynamically in notebook

# Initialize TinyLlama
model = OllamaLLM(model="tinyllama", base_url="http://host.docker.internal:11434")

# Prompt template emphasizing RAG and ignoring irrelevant content
template = """
You are a helpful assistant analyzing news articles.

Some articles may contain unreadable text or security blocks. Ignore any text that is not meaningful content.

You MUST answer using ONLY the context provided below. Do not use your own knowledge or general information.

Context from retrieved news articles:
{context}

Question:
{question}

Instructions:
- Only provide the title(s) of the relevant article(s) from the context.
- Give a concise summary (2-4 sentences) of the relevant information.
- If the context does not mention the requested word or topic, respond: "No relevant information found in the provided articles."
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Retriever function
def retriever(question, word=None, k=3):
    """
    Retrieves top-k relevant chunks from FAISS.
    If 'word' is provided, only include chunks that contain the word.
    """
    results = vector_db.similarity_search(question, k=10)  # retrieve more initially
    if word:
        results = [r for r in results if word.lower() in r.page_content.lower()]
    # limit to top-k after filtering
    results = results[:k]
    combined_context = "\n\n".join([r.page_content for r in results])
    return combined_context


# Chat function for notebook
def ask_question(question, word=None, k=3):
    context = retriever(question, word=word, k=k)
    response = chain.invoke({"context": context, "question": question})
    return response
