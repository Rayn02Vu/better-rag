from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from llm.LLM import LLM 
llm = LLM()

query_expansion_chain = PromptTemplate(
            input_variables=["question"],
            template="""Bạn là một trợ lý tạo truy vấn. 
            Nhiệm vụ của bạn là tạo ra 4 truy vấn tìm kiếm khác nhau từ câu hỏi người dùng được cung cấp,
            nhằm mở rộng và làm rõ chủ đề, nội dung mà người dùng muốn biết. 
            Đặt mỗi truy vấn mới trên một dòng riêng biệt. Cần tường minh rõ ràng và ngắn gọn, 
            chứa các từ khoá mà bạn cho là quan trọng liên quan đến đề tài câu hỏi gốc.
            Câu hỏi gốc: {question}"""
    )| llm.model | StrOutputParser()


query_logic_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là một trợ lý phân tích truy vấn thông minh.
    Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và đưa ra một hoặc nhiều truy vấn tìm kiếm TỐI ƯU cho hệ thống retrieval.
    Bạn cũng cần xác định bất kỳ metadata lọc nào (ví dụ: số chương, tiêu đề chương) nếu câu hỏi có nhắc đến.
    Nếu câu hỏi có vẻ chỉ là tổng quan và không cần tìm kiếm sâu, bạn có thể chỉ ra điều đó.

    Định dạng đầu ra là JSON như sau:
    {{
        "search_queries": ["query 1", "query 2"],
        "metadata_filters": {{ "chapter_number": "I", "section_title": "Tình hình quốc tế" }}, (tùy chọn)
        "needs_search": true/false,
        "reasoning": "Lý do phân tích của bạn"
    }}

    Ví dụ:
    Câu hỏi: "Nội dung chính chương 1 là gì?"
    Output: {{ "search_queries": ["nội dung chính chương 1", "chương 1 bối cảnh lịch sử", "tóm tắt chương 1"], "metadata_filters": {{ "chapter_number": "I" }}, "needs_search": true, "reasoning": "Câu hỏi yêu cầu nội dung cụ thể của chương 1." }}

    Câu hỏi: "Chào bạn"
    Output: {{ "search_queries": [], "metadata_filters": {{}}, "needs_search": false, "reasoning": "Đây là lời chào, không cần tìm kiếm." }}

    Câu hỏi: "Ai là chủ tịch Hồ Chí Minh?" (Giả sử thông tin này không có trong sách và LLM biết)
    Output: {{ "search_queries": ["chủ tịch Hồ Chí Minh"], "metadata_filters": {{}}, "needs_search": true, "reasoning": "Cần tìm kiếm thông tin về Hồ Chí Minh." }}
    """
    ),
    ("human", "{question}")
])


query_logic_chain = query_logic_prompt | llm.model.bind(response_format={"type": "json_object"}) | StrOutputParser()



review_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là một trợ lý đánh giá tài liệu.
    Nhiệm vụ của bạn là xem xét câu hỏi gốc của người dùng và các tài liệu được truy xuất.
    Đánh giá xem các tài liệu này có đủ để trả lời câu hỏi một cách đầy đủ và chính xác hay không.

    Định dạng đầu ra là JSON như sau:
    {{
        "assessment": "adequate/inadequate/needs_more_specific_search",
        "reasoning": "Lý do đánh giá của bạn (ví dụ: 'Các tài liệu thiếu thông tin về X', 'Các tài liệu rất phù hợp').",
        "suggested_new_search_queries": ["query bổ sung 1", "query bổ sung 2"] (tùy chọn, nếu assessment là 'needs_more_specific_search')
    }}

    Câu hỏi gốc: {original_question}
    Các tài liệu được truy xuất (chỉ nội dung và metadata quan trọng): {retrieved_documents}
    """
    ),
    ("human", "Hãy đánh giá các tài liệu trên cho câu hỏi '{original_question}'.")
])

review_agent_chain = review_agent_prompt | llm.model.bind(response_format={"type": "json_object"}) | StrOutputParser()


final_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là một trợ lý trả lời câu hỏi chuyên nghiệp.
    Sử dụng các tài liệu được cung cấp để trả lời câu hỏi của người dùng một cách đầy đủ, chính xác, và rõ ràng.
    Nếu câu hỏi không thể trả lời từ các tài liệu, hãy nói rằng bạn không có đủ thông tin.
    Đảm bảo câu trả lời của bạn có cấu trúc tốt và dễ hiểu. Trích dẫn số trang hoặc chương nếu có thể.

    Câu hỏi gốc: {original_question}
    Các tài liệu đã được kiểm tra và đánh giá: {context}
    """
    ),
    ("human", "{original_question}")
])

# Chain tạo câu trả lời cuối cùng
final_answer_chain = final_answer_prompt | llm.model | StrOutputParser()