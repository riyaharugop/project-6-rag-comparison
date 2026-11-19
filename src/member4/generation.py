
"""Person 4 - Gemini-based RAG answer generation."""

import os
from pathlib import Path
from time import perf_counter, sleep

from google import genai
from google.genai import errors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "gemini-3.1-flash-lite"


def load_api_key():
    """Read the Gemini API key from environment or the project's .env file."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        env_file = PROJECT_ROOT / ".env"

        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to the project .env file."
        )

    return api_key


def generate_answer(question, retrieved_chunks, model_name=MODEL_NAME):
    """Generate an answer using the question and retrieved RAG context."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant information to answer this question.",
            "sources": [],
            "latency_seconds": 0.0,
        }

    context_parts = []
    sources = []

    for index, chunk in enumerate(retrieved_chunks, start=1):

        if isinstance(chunk, str):
            text = chunk
            source = f"Retrieved chunk {index}"

        elif isinstance(chunk, dict):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            source = (
                chunk.get("source")
                or chunk.get("filename")
                or metadata.get("source")
                or metadata.get("filename")
                or chunk.get("chunk_id")
                or f"Retrieved chunk {index}"
            )

        else:
            raise TypeError(
                "Each retrieved chunk must be a dictionary or string."
            )

        if not isinstance(text, str) or not text.strip():
            continue

        context_parts.append(
            f"[Source {index}: {source}]\n{text}"
        )

        sources.append(str(source))

    if not context_parts:
        raise ValueError("Retrieved chunks contain no usable text.")

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a question-answering assistant for technical documentation.

Answer the user's question using ONLY the information in the
provided retrieved context.

Instructions:
1. Give a clear and accurate answer.
2. Do not invent facts or use outside knowledge.
3. If the context does not contain enough information, explicitly
   state that the information is insufficient.
4. Keep the answer relevant to the question.
5. Refer to the source labels when appropriate.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    client = genai.Client(api_key=load_api_key())
    start_time = perf_counter()

    try:
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "max_output_tokens": 512,
                        "temperature": 0.2,
                    },
                )
                break

            except errors.ClientError as exc:
                # Do not retry quota, authentication or other client errors.
                if getattr(exc, "code", None) == 429:
                    raise

                raise

            except errors.ServerError as exc:
                if (
                    getattr(exc, "code", None) != 503
                    or attempt == max_retries - 1
                ):
                    raise

                wait_time = 5 * (attempt + 1)

                print(
                    f"Gemini is temporarily unavailable. "
                    f"Retrying in {wait_time} seconds..."
                )

                sleep(wait_time)

        latency = perf_counter() - start_time
        answer = response.text

        if not answer:
            raise RuntimeError("Gemini returned an empty response.")

        return {
            "answer": answer.strip(),
            "sources": sources,
            "latency_seconds": round(latency, 4),
        }

    finally:
        client.close()
