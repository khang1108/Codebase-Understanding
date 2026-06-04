from domain.port.llm_port import LLMPort
from rag.prompt_templates import SYSTEEM_PROMT, build_prompt



class AnswerGenerator:
    def __init__(self, llm_port: LLMPort):
        self.llm_port = llm_port
    def generate_answer(self, question: str, context: list[dict]) -> str:
        """3/6/2026 Question+context -> prompt -> answer
        Function dùng để tạo câu trả lời dựa trên câu hỏi và context đã được xây dựng"""
        prompt = build_prompt(question, context)
        answer = self.llm_port.generate(prompt,system_prompt=SYSTEEM_PROMT)
        return answer