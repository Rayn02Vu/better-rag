import streamlit as st
from streamlit import session_state as ss
from llm.Agent import Agent
from llm.Tools import get_time_tool, get_weather_tool, retrieval_tool
from Indexing import get_vectorstore, embedding_model


st.title("Query Index")
st.sidebar.markdown(
    """
    # Query Index
    Tìm kiếm các chunks "phù hợp" với câu truy vấn.
    Bản chất là sử dụng so sánh khoảng cách các vector embedding.
    """
)

index = get_vectorstore("Novel")


queries = [
    "Trần Phi Vũ",
    "20",
    "ra lệnh",
    "phép thuật"
]

if prompt := st.text_input("Query from 'Novel'"):
    with st.spinner("Thinking..."):
        st.markdown("### Embeddings")
        embeddings = embedding_model.embed_query(prompt)
        st.write(embeddings)

    with st.spinner("Thinking..."):
        st.markdown("### Found Documents")

        search_results = index.similarity_search_with_score(prompt, k=3)

        for doc, score in search_results:
            st.write(f"Score: {score}")
            st.write(doc.page_content)
            st.write("---")

        
    

