import requests 
from domain.port.llm_port import LLMPort
class OllamaLLM(LLMPort):
    def __init__(self, model: str, ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.url = ollama_url

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """3/6/2026 Function dùng để gửi prompt đến Ollama và nhận về câu trả lời."""
        message=[]
        if system_prompt:
            message.append({"role": "system", 
                            "content": system_prompt})
        message.append({"role": "user", 
                        "content": prompt})
        payload = {
            "model": self.model,
            "messages": message,
            "stream": False,
            }   
        try:
            response = requests.post(
                f"{self.url}/api/chat",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except Exception as e:
            print(f"Error in OllamaLLM.generate: {e}")
            return "Sorry, I couldn't generate an answer at this time."
    def alive(self) -> bool:
        """3/6/2026 Function dùng để kiểm tra xem mô hình có còn sống không bằng cách gửi một yêu cầu đơn giản."""
        try:
            response = requests.get(f"{self.url}/api/models/{self.model}", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Error in OllamaLLM.alive: {e}")
            return False