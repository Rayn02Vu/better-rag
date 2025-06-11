from collections import defaultdict
from langchain_core.documents import Document


# --- Thuật toán Reciprocal Rank Fusion (RRF) ---
def reciprocal_rank_fusion(
    ranked_lists,
    k=60
):
    fused_scores = defaultdict(float)
    document_map = {} 

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_key = (doc.page_content, tuple(sorted(doc.metadata.items())))
            document_map[doc_key] = doc
            fused_scores[doc_key] += 1 / (k + rank + 1) 

    reranked_docs_with_scores = sorted(
        [(document_map[doc_key], score) for doc_key, score in fused_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
    return reranked_docs_with_scores


def meta_docs(documents: list[Document]) -> list[Document]:
    """
    Concatenate metadata to page content.
    """
    for doc in documents:
        doc.page_content = "\n".join(f"{k}: {v}" for k, v in doc.metadata.items()) + "\n" + doc.page_content
    return documents
    

def clean_text(text: str) -> str:
    import re
    """
    Thực hiện làm sạch cơ bản cho văn bản:
    - Loại bỏ nhiều khoảng trắng thành một.
    - Loại bỏ các ký tự đặc biệt không mong muốn (ví dụ: dấu gạch ngang lặp lại, dấu chấm lẻ).
    - Cắt bỏ khoảng trắng ở đầu/cuối.
    """
    text = re.sub(r'\s+', ' ', text)  # Thay thế nhiều khoảng trắng (bao gồm newline) bằng một khoảng trắng
    text = re.sub(r'[-–—_]{2,}', '', text) # Loại bỏ các dấu gạch ngang lặp lại (header/footer)
    text = re.sub(r'[•●■▪]', '', text) # Loại bỏ các ký hiệu bullet point lẻ
    text = re.sub(r'\.{2,}', '', text) # Loại bỏ nhiều dấu chấm liên tiếp
    text = re.sub(r'\[\d+\]', '', text) # Loại bỏ các số trong ngoặc vuông (thường là tham chiếu)
    text = text.strip()
    return text
    