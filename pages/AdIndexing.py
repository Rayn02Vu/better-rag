import streamlit as st
from streamlit import session_state as ss
from Indexing import get_vectorstore, setup_advanced_vectorstore

st.title("Advanced Indexing")
st.sidebar.markdown(
    """
    # Indexing
    Sử dụng regex để nhận diện các section trong tài liệu và thêm vào metadata.
    Metadata giúp LLM hiểu cấu trúc tài liệu và truy vấn nhanh chóng, chính xác hơn.
    """
)

index = get_vectorstore("VN-History")

if clicked := st.button("Load Vectorstore", type="primary", disabled=ss.loading):
    ss.loading = True
    st.rerun()

if ss.loading:
    st.markdown("### Vectorstore: 'VN-History'")
    st.markdown(f"Number of chunks: {len(index.docstore._dict)}")
    st.markdown(f"Source: data/VN-History.pdf")

    st.write("---")
    if hasattr(index.docstore, "_dict"):
        for id, doc in index.docstore._dict.items():
            st.write(f"##### Chunk: {id}")
            st.text(doc.page_content)
            st.divider()
