from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from rag.retriever import search


# ============================================================
# MEDIRAG RAG PIPELINE
# ============================================================

MODEL_NAME = "qwen3:4b"
TOP_K = 1


# ============================================================
# 1. LANGCHAIN + OLLAMA MODEL
# ============================================================

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
    num_predict=120,
    think=False,
)

output_parser = StrOutputParser()


# ============================================================
# 2. LANGCHAIN RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are MEDIRAG, a medical information assistant.

Answer the user's question using ONLY the retrieved MedQuAD
information provided below.

STRICT RULES:

1. Use only information contained in the retrieved sources.
2. Do not use outside medical knowledge.
3. Do not invent facts.
4. Do not combine unrelated diseases.
5. Answer the user's exact question.
6. Do not diagnose the user.
7. Do not provide personalized treatment.
8. Keep the answer short, clear, and easy to understand.
9. Prefer the source that directly matches the user's question.
10. Do not expose your reasoning or thinking process.
11. Return only the final answer.

If the retrieved information is insufficient, return exactly:

Sufficient information was not found in the retrieved medical sources.

At the end add:

This information is for educational purposes only and is not
a substitute for professional medical advice.

RETRIEVED MEDICAL INFORMATION:

{context}
""",
        ),
        (
            "human",
            """
User question:
{question}

Answer using ONLY the retrieved medical information.
""",
        ),
    ]
)


# ============================================================
# 3. LANGCHAIN CHAIN
# ============================================================

rag_chain = prompt | llm | output_parser


# ============================================================
# 4. RETRIEVE MEDICAL CONTEXT
# ============================================================

def retrieve_context(question: str, top_k: int = TOP_K):

    print("\n[MEDIRAG] Retrieving medical sources...")

    results = search(question)

    results = results[:top_k]

    print(f"[MEDIRAG] Retrieved {len(results)} source(s)")

    for i, result in enumerate(results, start=1):
        print(
            f"[MEDIRAG] Source {i}: "
            f"{result.get('question', '')} | "
            f"{result.get('source', '')} | "
            f"score={result.get('score', 0):.4f}"
        )

    return results


# ============================================================
# 5. FORMAT CONTEXT
# ============================================================

def format_context(results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        question = result.get("question", "")
        answer = result.get("answer", "")[:3000]
        source = result.get("source", "")
        url = result.get("url", "")

        context_parts.append(
            f"""
SOURCE {i}

Question:
{question}

Answer:
{answer}

Source:
{source}

URL:
{url}
""".strip()
        )

    return "\n\n".join(context_parts)


# ============================================================
# 6. GENERATE GROUNDED ANSWER
# ============================================================

def generate_answer(question: str, context: str):

    print("[MEDIRAG] Sending context through LangChain → Qwen3...")

    response = rag_chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    answer = response.strip()

    # Defensive cleanup if an older Qwen response includes </think>
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1].strip()

    if answer.startswith("<think>"):
        answer = answer.replace("<think>", "", 1).strip()

    print("[MEDIRAG] Qwen3 response received")

    return answer


# ============================================================
# 7. COMPLETE RAG PIPELINE
# ============================================================

def ask_medical_question(question: str, top_k: int = TOP_K):

    print("\n" + "=" * 70)
    print("[MEDIRAG] NEW QUESTION")
    print(question)
    print("=" * 70)

    # Step 1: retrieve
    results = retrieve_context(question, top_k)

    # Step 2: build context
    context = format_context(results)

    # Step 3: LangChain → Qwen3
    answer = generate_answer(question, context)

    return {
        "question": question,
        "answer": answer,
        "results": results,
    }


# ============================================================
# 8. DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_question = "What are the symptoms of diabetes?"

    result = ask_medical_question(test_question)

    print("\n" + "=" * 70)
    print("GENERATED MEDIRAG ANSWER")
    print("=" * 70)
    print(result["answer"])

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for i, source in enumerate(result["results"], start=1):
        print(
            f"{i}. {source.get('question', '')} | "
            f"{source.get('source', '')} | "
            f"{source.get('url', '')}"
        )