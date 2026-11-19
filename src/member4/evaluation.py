
"""Person 4 - Evaluation metrics for the RAG comparison project."""

import re
from time import perf_counter


def _get_text(chunk):
    """Extract text from a retrieved chunk."""
    if isinstance(chunk, str):
        return chunk

    if isinstance(chunk, dict):
        return chunk.get("text", "")

    return ""


def _get_tokens(text):
    """Convert text into a set of lowercase words."""
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def context_recall(reference_evidence, retrieved_chunks):
    """
    Measure how much reference evidence is present in retrieved context.

    Formula:
        Reference evidence tokens found in context /
        Total reference evidence tokens
    """

    if isinstance(reference_evidence, str):
        reference_evidence = [reference_evidence]

    reference_text = " ".join(reference_evidence)
    reference_tokens = _get_tokens(reference_text)

    if not reference_tokens:
        return 0.0

    retrieved_text = " ".join(
        _get_text(chunk) for chunk in retrieved_chunks
    )

    retrieved_tokens = _get_tokens(retrieved_text)

    matched_tokens = reference_tokens.intersection(retrieved_tokens)

    return round(len(matched_tokens) / len(reference_tokens), 4)


def context_precision(reference_evidence, retrieved_chunks):
    """
    Measure the proportion of retrieved chunks relevant to the
    reference evidence.

    A chunk is considered relevant when at least 15% of its
    unique words overlap with the reference evidence.
    """

    if isinstance(reference_evidence, str):
        reference_evidence = [reference_evidence]

    reference_tokens = _get_tokens(" ".join(reference_evidence))

    if not reference_tokens or not retrieved_chunks:
        return 0.0

    relevant_count = 0

    for chunk in retrieved_chunks:
        chunk_tokens = _get_tokens(_get_text(chunk))

        if not chunk_tokens:
            continue

        overlap = chunk_tokens.intersection(reference_tokens)

        relevance = len(overlap) / len(chunk_tokens)

        if relevance >= 0.15:
            relevant_count += 1

    return round(relevant_count / len(retrieved_chunks), 4)


def groundedness(answer, retrieved_chunks):
    """
    Estimate how well generated answer sentences are supported
    by retrieved context.

    A sentence is considered supported when at least 30% of
    its unique words occur in the retrieved context.

    This is a lexical approximation, not an official RAGAS score.
    """

    if not answer or not retrieved_chunks:
        return 0.0

    context = " ".join(
        _get_text(chunk) for chunk in retrieved_chunks
    )

    context_tokens = _get_tokens(context)

    if not context_tokens:
        return 0.0

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())

    valid_sentences = [
        sentence for sentence in sentences
        if _get_tokens(sentence)
    ]

    if not valid_sentences:
        return 0.0

    supported_count = 0

    for sentence in valid_sentences:
        sentence_tokens = _get_tokens(sentence)

        overlap = sentence_tokens.intersection(context_tokens)

        support_ratio = len(overlap) / len(sentence_tokens)

        if support_ratio >= 0.30:
            supported_count += 1

    return round(supported_count / len(valid_sentences), 4)


def calculate_latency(retrieval_latency, generation_latency):
    """
    Calculate total end-to-end latency in seconds.
    """

    return round(
        float(retrieval_latency) + float(generation_latency),
        4
    )


def evaluate_response(
    reference_evidence,
    retrieved_chunks,
    answer,
    retrieval_latency=0.0,
    generation_latency=0.0
):
    """
    Calculate all evaluation metrics for one question.

    Returns a dictionary suitable for storing in results.csv.
    """

    return {
        "context_recall": context_recall(
            reference_evidence,
            retrieved_chunks
        ),

        "context_precision": context_precision(
            reference_evidence,
            retrieved_chunks
        ),

        "groundedness": groundedness(
            answer,
            retrieved_chunks
        ),

        "retrieval_latency_seconds": round(
            float(retrieval_latency), 4
        ),

        "generation_latency_seconds": round(
            float(generation_latency), 4
        ),

        "end_to_end_latency_seconds": calculate_latency(
            retrieval_latency,
            generation_latency
        )
    }
