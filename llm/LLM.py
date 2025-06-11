import streamlit as st
from streamlit import session_state as ss
from langchain_openai import ChatOpenAI


api_key = st.secrets.get("CONO_API_KEY", None)

class LLM:
    model = ChatOpenAI(
        model="cono-3-exp",
        api_key=api_key,
        base_url="https://api.arcanic.ai/v1",
        temperature=0.5,
    )

    bind = model.bind

    def __init__(
            self, 
            system_prompt: str = None, 
        ):

        self.system_prompt = system_prompt or (
            "Bạn hãy đóng vai một LLM thân thiện"
            "Hãy trả lời câu hỏi người dùng dựa trên kiến thức của bạn và các tài liệu tìm kiếm được."
            "Câu trả lời chính xác và đúng trọng tâm. Chỉ sử dụng những tài liệu nhằm trả lời câu hỏi người dùng"
            "Hãy cố gắng suy nghĩ xem người dùng muốn biết thông tin như thế nào. Tổng hợp để trả lời khách quan nhất."
        )

    def invoke(self, query: str, context: str | list[str] | None = None) -> str:
        if isinstance(context, list):
            context = "\n".join(context)

        messages = [
            ("system", self.system_prompt),
            ("human", query)
        ] 
        if context:
            messages.append(("human", f"Có thể tham khảo tài liệu sau nếu cảm thấy cần thiết:\n {context}"))
        respone = self.model.invoke(messages)
        return respone.content
