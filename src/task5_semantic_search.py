"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""
import chromadb
from sentence_transformers import SentenceTransformer

from task4_chunking_indexing import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

# def semantic_search(query: str, top_k: int = 10) -> list[dict]:
#     """
#     Tìm kiếm ngữ nghĩa sử dụng vector similarity.

#     Args:
#         query: Câu truy vấn
#         top_k: Số lượng kết quả tối đa

#     Returns:
#         List of {
#             'content': str,      # Nội dung chunk
#             'score': float,      # Cosine similarity score
#             'metadata': dict     # source, doc_type, chunk_index
#         }
#         Sorted by score descending.
#     """
#     # TODO: Implement semantic search
#     #
#     # Bước 1: Embed query bằng cùng model ở Task 4
#     # Bước 2: Query vector store (cosine similarity)
#     # Bước 3: Return top_k results
#     #
#     # Ví dụ với Weaviate:
#     # import weaviate
#     # from sentence_transformers import SentenceTransformer
#     #
#     # model = SentenceTransformer("BAAI/bge-m3")
#     # query_embedding = model.encode(query).tolist()
#     #
#     # client = weaviate.connect_to_local()
#     # collection = client.collections.get("DrugLawDocs")
#     #
#     # results = collection.query.near_vector(
#     #     near_vector=query_embedding,
#     #     limit=top_k,
#     #     return_metadata=MetadataQuery(distance=True)
#     # )
#     #
#     # return [
#     #     {
#     #         "content": obj.properties["content"],
#     #         "score": 1 - obj.metadata.distance,  # distance → similarity
#     #         "metadata": {"source": obj.properties["source"], ...}
#     #     }
#     #     for obj in results.objects
#     # ]
#     raise NotImplementedError("Implement semantic_search")

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa bằng ChromaDB.

    ChromaDB lưu vector từ Task 4. Ở đây ta embed query bằng đúng
    EMBEDDING_MODEL đã dùng khi index, sau đó query bằng query_embeddings.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("Chưa tìm thấy ChromaDB collection. Hãy chạy Task 4 trước.")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output = []
    for content, metadata, distance in zip(documents, metadatas, distances):
        # Vì Task 4 tạo collection với cosine distance:
        # distance càng nhỏ càng giống, nên đổi sang similarity score.
        score = 1 - distance

        output.append({
            "content": content,
            "score": float(score),
            "metadata": metadata or {},
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]

if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
