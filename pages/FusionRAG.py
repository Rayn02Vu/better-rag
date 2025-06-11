import streamlit as st

from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from Indexing import get_vectorstore
from llm.LLM import LLM

index = get_vectorstore("VN-History")

llm = LLM()

vt_retriever = index.as_retriever()
bm_retriever = BM25Retriever.from_documents([doc for doc in index.docstore._dict.values()])
bm_retriever.k = 3

hybrid_retriever = EnsembleRetriever(
    retrievers=[vt_retriever, bm_retriever],
    weights=[0.6, 0.4]
)

st.title("Fusion RAG")


if prompt := st.text_input("Query from 'LichsuDang.pdf'"):

    st.divider()
    from llm.Chain import query_expansion_chain
    with st.spinner("Đang tạo các truy vấn mở rộng..."):
        expanded_queries_raw = query_expansion_chain.invoke({"question": prompt})
        expanded_queries = [q.strip() for q in expanded_queries_raw.split('\n') if q.strip()]
        if prompt not in expanded_queries:
            expanded_queries.insert(0, prompt)
        st.subheader("Các truy vấn đã được mở rộng:")
        for i, q in enumerate(expanded_queries):
            st.write(f"- `{q}`")

    st.divider()


    all_retrieved_docs_map = []
    with st.expander("Xem kết quả truy xuất cho từng truy vấn"):
        for i, query_to_search in enumerate(expanded_queries):
            st.subheader(f"🔍 `{query_to_search}`")
            
            with st.spinner(f"Đang thực hiện Hybrid Search cho '{query_to_search}'..."):
                hybrid_docs = hybrid_retriever.get_relevant_documents(query_to_search)
                st.markdown(f"**Kết quả Hybrid Search ({len(hybrid_docs)} docs):**")
                for doc in hybrid_docs:
                    st.write(f"- `{doc.page_content[:100].replace('\n', ' ')}...`")
                    all_retrieved_docs_map.append(hybrid_docs) 
            st.divider()

    st.success(f"Tổng số tài liệu được truy xuất: **{len(all_retrieved_docs_map)}**")
    st.divider()


    from llm.utils import reciprocal_rank_fusion
    with st.spinner("Đang hợp nhất và xếp hạng lại các tài liệu..."):
        reranked_docs = reciprocal_rank_fusion(all_retrieved_docs_map)
        st.subheader("Top 4 tài liệu đã được hợp nhất và xếp hạng lại bằng RRF:")
        for i, (doc, score) in enumerate(reranked_docs[:5]):
            st.write(f"**{i+1}.** {score:.4f} - `{doc.page_content[:150].replace('\n', ' ')}...`")
    st.divider()

    context_docs = reranked_docs[:4]
    context = "\n\n".join([doc.page_content for doc, _ in context_docs])
    with st.spinner("Đạng tạo câu trả lời..."):
        response = llm.invoke(prompt, context)
        st.subheader("Câu trả lời cuối cùng:")
        st.write(response)

        
