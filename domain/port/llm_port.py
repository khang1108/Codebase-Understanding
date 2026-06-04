from abc import ABC, abstractmethod
class LLMPort(ABC):
    @abstractmethod
    # 3/6/2026 Generate câu trả lời dựa trên prompt đã xây dựng
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        pass

    @abstractmethod
    # 3/6/2026 Check mô hình còn sống không
    def alive(self) -> bool:
        pass