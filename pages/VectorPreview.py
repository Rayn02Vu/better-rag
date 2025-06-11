import streamlit as st
from streamlit import session_state as ss
from Indexing import get_vectorstore, embedding_model


st.title("Indexing")
st.sidebar.markdown(
    """
    # Vector Preview
    - Embedding: biến đổi dữ liệu text thành các vector ngữ nghĩa.
    - Chunking: chia tài liệu thành các đoạn nhỏ.
    - Vector store: cơ sở dữ liệu lưu trữ các vector chunks.
    """
)

index = get_vectorstore("Novel")

if clicked := st.button("Load Vectorstore", type="primary", disabled=ss.loading):
    ss.loading = True
    st.rerun()

if ss.loading:
    st.markdown("### Vectorstore: 'Novel'")
    st.markdown(f"Number of chunks: {len(index.docstore._dict)}")
    st.markdown(f"Source: data/TestData.txt")

    st.write("---")
    st.markdown("### Chunks")
    if hasattr(index.docstore, "_dict"):
        for id, doc in index.docstore._dict.items():
            st.write("ID: ", id)
            st.text(doc.page_content)
            st.write("---")


        
    

