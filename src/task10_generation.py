"""
Task 10 - Generation with citations.

Steps:
    1. Retrieve relevant chunks from Task 9.
    2. Reorder chunks to reduce "lost in the middle".
    3. Format context with citation metadata.
    4. Ask the LLM to answer with citations.
    5. Return an unverifiable answer when evidence is insufficient.
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.task9_retrieval_pipeline import retrieve
except ModuleNotFoundError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION
# =============================================================================

# top_k: 5 chunks gives enough evidence without making the prompt too long.
TOP_K = 5

# top_p: 0.9 keeps the answer natural while still grounded in context.
TOP_P = 0.9

# temperature: 0.3 is low because RAG answers should be factual.
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the question comprehensively in Vietnamese.

Use only the provided context. For every factual claim, add a citation
immediately after that claim.

Citation format:
- Every citation must be a bracketed citation containing a real source name,
  then a comma, then a four-digit year.
- Use the "Citation hint" in each context block whenever possible.
- If no hint is available, infer the real source name and year from metadata,
  source filename, title, URL, and content.
- Never output placeholder citations using generic words like Nguon, Source,
  Unknown, or Year.
- Do not cite article numbers as the citation. If an article number matters,
  mention it in the sentence text, then cite the real source and year.

If the context does not explicitly support the answer, say:
"Toi khong the xac minh thong tin nay tu nguon hien co".

Rules:
- Do not guess.
- Do not use information outside the provided context.
- Every factual claim must have a citation.
- Structure the answer with clear paragraphs."""


# =============================================================================
# CITATION HELPERS
# =============================================================================

def _metadata_text(metadata: dict, content: str) -> str:
    values = [str(v) for v in metadata.values() if v is not None]
    values.append(content[:1000])
    return " ".join(values)


def infer_year(metadata: dict, content: str) -> str:
    """Infer a 4-digit year from metadata, filename, URL, or content."""
    text = _metadata_text(metadata, content)
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    return years[-1] if years else "n.d."


def clean_source_name(raw_source: str) -> str:
    """Turn a source filename into a readable source name without hard-coding docs."""
    source = Path(raw_source).stem if raw_source else "unknown-source"
    source = re.sub(r"\b(?:19|20)\d{2}\b", "", source)
    source = source.replace("_", " ").replace("-", " ")
    source = re.sub(r"\s+", " ", source).strip()
    return source or raw_source or "unknown-source"


def build_citation_hint(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    content = chunk.get("content", "")
    raw_source = (
        metadata.get("source")
        or metadata.get("filename")
        or metadata.get("url")
        or "unknown-source"
    )
    source_name = clean_source_name(str(raw_source))
    year = infer_year(metadata, content)
    return f"[{source_name}, {year}]"


def fix_placeholder_citations(answer: str, chunks: list[dict]) -> str:
    """
    Replace generic placeholder citations if the model accidentally emits them.

    The replacement is still derived from retrieved metadata via citation hints.
    """
    if not chunks:
        return answer

    fallback_citation = build_citation_hint(chunks[0])
    placeholder_patterns = [
        r"\[(?:Nguon|Nguồn)\s*,\s*(?:Nam|Năm|\d{4}|n\.d\.)\]",
        r"\[(?:Source|Unknown)\s*,\s*(?:Year|\d{4}|N/A|n\.d\.)\]",
    ]

    fixed = answer
    for pattern in placeholder_patterns:
        fixed = re.sub(pattern, fallback_citation, fixed, flags=re.IGNORECASE)
    return fixed


# =============================================================================
# DOCUMENT REORDERING
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to reduce the "lost in the middle" effect.

    Input order by score: [1, 2, 3, 4, 5]
    Output order:         [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks

    reordered = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])

    tail = []
    for i in range(1, len(chunks), 2):
        tail.append(chunks[i])
    reordered.extend(reversed(tail))

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks into context blocks.

    Each block includes a citation hint in the required [source, year] format.
    The hint is inferred from metadata and filenames, not hard-coded per document.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = (
            metadata.get("source")
            or metadata.get("filename")
            or metadata.get("url")
            or f"source_{i}"
        )
        doc_type = (
            metadata.get("type")
            or metadata.get("doc_type")
            or chunk.get("source", "unknown")
        )
        page_index = metadata.get("page_index")
        page_text = f"\nPage index: {page_index}" if page_index is not None else ""
        score = chunk.get("score", 0.0)
        citation_hint = build_citation_hint(chunk)

        context_parts.append(
            f"Context block {i}\n"
            f"Source metadata: {source}\n"
            f"Document type: {doc_type}"
            f"{page_text}\n"
            f"Citation hint: {citation_hint}\n"
            f"Retrieval score: {score:.3f}\n"
            f"Content:\n{chunk.get('content', '')}\n"
        )

    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation with citations.

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str
        }
    """
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Toi khong the xac minh thong tin nay tu nguon hien co",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    user_message = f"""Context:
{context}

---

Question: {query}

Remember: citations must be bracketed as source name, comma, year. Use the
Citation hint values from the context when they are available."""

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        citation = build_citation_hint(chunks[0])
        return {
            "answer": f"Mock LLM answer based on retrieved context. {citation}",
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid"),
        }

    try:
        from openai import OpenAI

        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        citation = build_citation_hint(chunks[0])
        answer = f"Mock LLM answer due to API error: {e}. {citation}"

    answer = fix_placeholder_citations(answer, chunks)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    test_queries = [
         "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
