from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.document_loaders import TextLoader
from llm.LLM import LLM 
llm = LLM()

query_expansion_chain = PromptTemplate(
            input_variables=["question"],
            template="""Bạn là một trợ lý tạo truy vấn. 
            Nhiệm vụ của bạn là tạo ra 3 truy vấn tìm kiếm khác nhau từ câu hỏi người dùng được cung cấp,
            nhằm mở rộng và làm rõ chủ đề, nội dung mà người dùng muốn biết. 
            Đặt mỗi truy vấn mới trên một dòng riêng biệt. Cần tường minh rõ ràng và ngắn gọn, 
            chứa các từ khoá mà bạn cho là quan trọng liên quan đến đề tài câu hỏi gốc.
            Câu hỏi gốc: {question}"""
    )| llm.model | StrOutputParser()