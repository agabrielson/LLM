#!/usr/bin/env python3

# === Usage Example ===

# Initialize QA system
qa_system = RetrievalQA(top_k=2)

# Query example
question = "tell me about colt firearms?"
answer = qa_system.query(question)

print("Answer:", answer)