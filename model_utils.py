from sentence_transformers import SentenceTransformer
import ollama

def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_llm_model():
    def local_llm(prompt):
        response = ollama.chat(model='phi3', messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    return local_llm
