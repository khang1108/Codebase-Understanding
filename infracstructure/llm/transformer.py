from domain.port.llm_port import LLMPort
 
 
class TransformerLLM(LLMPort):
    """
    Local LLM backend dùng HuggingFace Transformers.
    Drop-in replacement cho OllamaLLM — cùng interface, cùng tên attribute.
 
    Cài đặt:
        pip install transformers torch accelerate
    """
 
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        max_new_tokens: int = 512,
        device: str = "auto",
    ):
        self.model = model                  # giống OllamaLLM.model
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._pipeline = None               # lazy load
 
    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
 
    def _load(self):
        """Load pipeline lần đầu tiên khi generate() được gọi."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except ImportError:
            raise RuntimeError(
                "transformers chưa được cài. Chạy: pip install transformers torch accelerate"
            )
        self._pipeline = pipeline(
            "text-generation",
            model=self.model,
            device_map=self._device,
            trust_remote_code=True,
        )
 
    # ------------------------------------------------------------------ #
    # LLMPort interface
    # ------------------------------------------------------------------ #
 
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Build chat messages rồi gọi pipeline, trả về answer string."""
        self._load()
 
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
 
        try:
            outputs = self._pipeline(
                messages,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        except Exception as e:
            raise RuntimeError(f"TransformerLLM generation failed: {e}")
 
        # HuggingFace chat pipeline trả về list of messages
        generated = outputs[0]["generated_text"]
        if isinstance(generated, list):
            assistant_msgs = [m for m in generated if m.get("role") == "assistant"]
            if assistant_msgs:
                return assistant_msgs[-1]["content"].strip()
            return generated[-1].get("content", "").strip()
 
        # text-generation thông thường trả về string
        return str(generated).strip()
 
    def is_available(self) -> bool:
        """Kiểm tra transformers + torch đã cài chưa (không load model)."""
        try:
            import transformers  
            import torch          
            return True
        except ImportError:
            return False

    def alive(self) -> bool:
        """Check if the Transformer backend is available for use."""
        return self.is_available()