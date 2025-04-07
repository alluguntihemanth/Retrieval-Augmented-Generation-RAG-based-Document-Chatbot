def generate_answer(query, context, llm):
    prompt = f"""
You are a helpful assistant. Answer the question using ONLY the context below.
Format your answer with structured markdown: bullet points, tables, or paragraphs.
If the answer isn't in the context, say "Not found in the document."

Context:
{context}

Question: {query}
Answer:
"""
    response = llm(prompt)
    return response
