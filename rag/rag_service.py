from domain.models.rag_answer import RagAnswer, SourceChunk
from domain.port.llm_port import LLMPort
from rag.context_builder import build_context, chunk_to_dict
from rag.answer_generator import AnswerGenerator

class RagService:
    def __init__(self, llm: LLMPort):
        self.llm = llm
        self.answer_generator = AnswerGenerator(llm)
    def answer_question(self, question: str, retrieved_chunks: list) -> RagAnswer:
        """3/6/2026 pipeline chính của RAG, nó nhận vào một câu hỏi và một danh sách các chunk đã được truy xuất,"""
        try:
            context = build_context(retrieved_chunks)
            answer_text = self.answer_generator.generate_answer(question, chunk_to_dict(context))
            rag_answer = RagAnswer(
                question=question,
                answer=answer_text,
                source_chunks=context,
                model_used=getattr(self.llm,"model","UnknownModel")
            )
            return rag_answer
        except Exception as e:
            # 3/6/2026 Xử lý lỗi nếu có
            print(f"Error in RagService: {e}")
            return RagAnswer(
                question=question,
                answer="Sorry, I couldn't generate an answer at this time.",
                source_chunks=[],
                model_used=self.answer_generator.llm_port.__class__.__name__
            )