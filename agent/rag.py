import faiss
import ollama
from sentence_transformers import SentenceTransformer


class RAG:
    def __init__(self, documents):
        # Create embeddings
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.encoder.encode(documents)

        # FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        self.documents = documents

    def retrieve(self, text):
        query_emb = self.encoder.encode([text])
        _, idx = self.index.search(query_emb, k=2)
        return [self.documents[i] for i in idx[0]]

    def explain(self, text):
        context = "\n".join(self.retrieve(text))

        prompt = f"""
You are an academic assistant.

Context from similar research papers:
{context}

Research Abstract:
{text}

Explain the sentiment in 3–4 academic sentences.
"""

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]



