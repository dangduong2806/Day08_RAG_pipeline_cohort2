"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

import time
import requests
import json

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

PAGEINDEX_API_BASE = "https://api.pageindex.ai"
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
DOC_IDS_FILE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    """
    Upload documents lên PageIndex và lưu doc_id vào data/pageindex_doc_ids.json.
    """
    if not PAGEINDEX_API_KEY:
        raise ValueError("Thiếu PAGEINDEX_API_KEY trong .env")
    
    uploaded = load_pageindex_doc_ids()
    uploaded_ids = {item["doc_id"] for item in uploaded}
    uploaded_files = {item["filename"] for item in uploaded}

    pdf_files = list((LANDING_DIR / "legal").glob("*.pdf"))
    if not pdf_files:
        print("Không tìm thấy PDF trong data/landing/legal/")
        return []

    uploaded = []

    for pdf_file in pdf_files:
        if pdf_file.name in uploaded_files:
            print(f"Skip uploaded: {pdf_file.name}")
            continue

        print(f"Uploading: {pdf_file.name}")

        with pdf_file.open("rb") as file:
            response = requests.post(
                f"{PAGEINDEX_API_BASE}/doc/",
                headers={"api_key": PAGEINDEX_API_KEY},
                files={"file": file},
                timeout=120,
            )
        
        print("Status:", response.status_code)
        print("Response:", response.text)

        if response.status_code == 403 and "LimitReached" in response.text:
            print("Đã chạm giới hạn upload PageIndex. Dừng upload và dùng các doc_id đã có.")
            break

        response.raise_for_status()
        data = response.json()

        uploaded.append({
            "doc_id": data["doc_id"],
            "filename": pdf_file.name,
        })

        print(f"  Uploaded: {pdf_file.name} -> {data['doc_id']}")

    DOC_IDS_FILE.write_text(
        json.dumps(uploaded, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return uploaded
   
def load_pageindex_doc_ids() -> list[dict]:
    if DOC_IDS_FILE.exists():
        return json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    return []

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    # TODO: Implement PageIndex query
    #
    """
    Vectorless retrieval sử dụng PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        raise ValueError("Thiếu PAGEINDEX_API_KEY trong .env")

    docs = load_pageindex_doc_ids()
    if not docs:
        raise ValueError("Chưa có doc_id. Hãy chạy upload_documents() trước.")

    results = []

    for doc in docs:
        doc_id = doc["doc_id"]

        response = requests.post(
            f"{PAGEINDEX_API_BASE}/retrieval/",
            headers={"api_key": PAGEINDEX_API_KEY},
            json={
                "doc_id": doc_id,
                "query": query,
                "thinking": False,
            },
            timeout=60,
        )
        response.raise_for_status()

        retrieval_id = response.json()["retrieval_id"]

        for _ in range(30):
            status_response = requests.get(
                f"{PAGEINDEX_API_BASE}/retrieval/{retrieval_id}/",
                headers={"api_key": PAGEINDEX_API_KEY},
                timeout=60,
            )
            status_response.raise_for_status()
            status_data = status_response.json()

            if status_data.get("status") == "completed":
                for node in status_data.get("retrieved_nodes", []):
                    for item in node.get("relevant_contents", []):
                        results.append({
                            "content": item.get("relevant_content", ""),
                            "score": 1.0,
                            "metadata": {
                                "doc_id": doc_id,
                                "filename": doc.get("filename", "unknown"),
                                "title": node.get("title", ""),
                                "node_id": node.get("node_id", ""),
                                "page_index": item.get("page_index"),
                            },
                            "source": "pageindex",
                        })
                break

            time.sleep(2)

    return results[:top_k]



if __name__ == "__main__":
    docs = load_pageindex_doc_ids()

    if not docs:
        print("Uploading documents...")
        docs = upload_documents()
    else:
        print(f"Using existing PageIndex doc ids: {len(docs)}")

    if not docs:
        print("Chưa có doc_id để query.")
    else:
        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
