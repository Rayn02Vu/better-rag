import streamlit as st
from streamlit import session_state as ss
from Indexing import get_vectorstore
from llm.LLM import LLM
st.title("Simple RAG")

vectorstore = get_vectorstore("VN-History")

retriever = vectorstore.as_retriever()

llm = LLM()

if prompt := st.text_input("Query from 'LichsuDang.pdf'"):
    docs =  retriever.invoke(prompt)

    st.markdown("### Retrieved Documents")
    
    for doc in docs:
        st.write("---")
        st.markdown("##### **Document**")
        st.text(doc.page_content)

    response = llm.invoke(prompt, [doc.page_content for doc in docs])

    st.markdown("### LLM Answer")
    st.write(response)


    