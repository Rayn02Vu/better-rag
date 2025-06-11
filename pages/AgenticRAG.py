import streamlit as st
from typing import List

from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate

from Indexing import get_vectorstore
from llm.Agent import Agent
from llm.Tools import retrieval_tool, search_tool

from llm.Chain import query_logic_chain, review_agent_chain, final_answer_chain


st.title("Agentic RAG")
st.sidebar.markdown(
    """
    # Agentic RAG
    Multiple query, retrieval and self-review.
    
    # Data
    *LichsuDang.pdf* <br>
    A document about the history of the Communist Party of Vietnam
    """,
    unsafe_allow_html=True
)

main_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Bạn là một trợ lý có khả năng truy cập nhiều công cụ. Bạn cần giải quyết một câu hỏi con được giao.
    Sử dụng các công cụ có sẵn để tìm kiếm thông tin cần thiết.
    Khi bạn tìm thấy thông tin phù hợp, hãy trả về kết quả đó dưới dạng một bản tóm tắt ngắn gọn và trung thực.
    Nếu không tìm thấy thông tin, hãy nói rõ.
    
    Các tool hiện có: 
    - search_tool: tìm kiếm trên web với danh sách các từ khóa
    - retriver_tool: tìm kiếm trong tài liệu với key='VN-History' và một câu query
    
    Ưu tiên sử dụng web_search đối với việc tìm kiếm các thông tin mới có tính cập nhật
    Sử dụng retrieval đối với thông tin kinh điển, mang tính học thuật.
    
    Lịch sử trò chuyện với người dùng:
    {chat_history}

    Câu hỏi con cần giải quyết: {input}
    {agent_scratchpad}
    """)
])

main_agent = Agent(tools=[retrieval_tool, search_tool], prompt_template=main_agent_prompt)

retriever = get_vectorstore("VN-History").as_retriever()

def self_query(search_queries: list[str]):
    retrieved_docs_list: List[Document] = []
    for q in search_queries:
        st.write(f"Tìm kiếm cho: `{q}`")
        docs_for_query = retriever.invoke(q) 
        st.markdown("##### 📚 Tài liệu được truy xuất:")
        if docs_for_query:
            for doc in docs_for_query:
                st.write(f"- Trang {doc.metadata.get('page_number', 'N/A')}, Chương: {doc.metadata.get('chapter_num', 'N/A')} - `{doc.page_content[:150]}...`")
            retrieved_docs_list.extend(docs_for_query)
        else:
            st.write("Không tìm thấy tài liệu nào.")
    unique_docs: List[Document] = []
    seen_content = set()
    for doc in retrieved_docs_list:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)
    context_for_review = "\n\n".join([f"Trang {d.metadata.get('page_number', 'N/A')}, Chương {d.metadata.get('chapter_number', 'N/A')}: {d.page_content}" for d in unique_docs[:10]])
    return context_for_review, unique_docs

from streamlit import session_state as ss

MAX_LOOP = 4

if prompt := st.text_input("Nhập câu hỏi của bạn về 'LichsuDang.pdf'"):
    ss.messages.append({"role": "user", "content": prompt})
    with st.spinner("Đang xử lý Agentic RAG..."):
        
        st.subheader("Phân tích truy vấn:")
        query_analysis_raw = query_logic_chain.invoke({"question": prompt})

        import json
        query_analysis: dict = json.loads(query_analysis_raw)
        search_queries = query_analysis.get("search_queries", [])
        metadata_filters = query_analysis.get("metadata_filters", {})
        needs_search = query_analysis.get("needs_search", True)
        for query in search_queries:
            st.write(f"- `{query}`")
                
        if not needs_search:
            response = query_analysis.get("reasoning", "Không có thông tin cụ thể để trả lời.")
            st.chat_message("ai").write(response)
            ss.messages.append({"role": "ai", "content": response})
            st.stop() 
        if not search_queries:
            search_queries = [prompt]
            
        
        loop = 0
        queries = search_queries
        context_for_review = ""
        unique_docs = []
        
        while loop < MAX_LOOP:
            st.divider()
            st.subheader("Thực hiện Truy xuất Tài liệu (Retrieval)")

            context_for_review, unique_docs = self_query(queries)
            
            st.subheader("Đánh giá Tài liệu:")
            review_analysis_raw = review_agent_chain.invoke({
                "original_question": prompt,
                "retrieved_documents": context_for_review
            })

            review_analysis: dict = json.loads(review_analysis_raw)
            
            assessment = review_analysis.get("assessment", "inadequate")
            suggested_new_search_queries = review_analysis.get("suggested_new_search_queries", [])
            st.write(f"**Kết quả đánh giá:** {review_analysis.get("reasoning", "")}")
            
            if assessment == "inadequate" and suggested_new_search_queries:
                st.warning("Tài liệu chưa đủ. Đang thử tìm kiếm bổ sung...")
                st.info(f"Tìm kiếm bổ sung: {suggested_new_search_queries}")
                queries = suggested_new_search_queries
            elif assessment == "adequate":
                st.success("Các tài liệu đã được đánh giá là đủ.")
                break
            else:
                st.warning("Đánh giá không xác định hoặc không có đề xuất tìm kiếm mới.")
                break
            loop += 1


        st.subheader("Tạo Câu trả lời Cuối cùng:")
        if not unique_docs:
            final_answer = "Xin lỗi, tôi không tìm thấy tài liệu liên quan nào để trả lời câu hỏi của bạn."
        else:
            final_answer = final_answer_chain.invoke({
                "original_question": prompt,
                "context": context_for_review
            })
        
        st.markdown("### Câu trả lời cuối cùng: ")
        st.write(final_answer)
