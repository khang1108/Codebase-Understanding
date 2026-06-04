from flask import Flask, request, jsonify
import os
import sys
import traceback
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from infracstructure.llm.ollama_llm import OllamaLLM
except ImportError as e:
    OllamaLLM = None
    print(f"[INFO] OllamaLLM import failed ({e}).")

from infracstructure.llm.transformer import TransformerLLM
from rag.rag_service import RagService

def create_app(
    ollama_url: str = "http://localhost:11434",
    model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    retrieval_service=None,
) -> Flask:
    app = Flask(__name__)

    # Default to Transformer backend; fallback to Ollama if transformer unavailable
    try:
        llm = TransformerLLM(model=model)
    except Exception as e:
        print(f"Transformer LLM unavailable ({e}), falling back to Ollama.")
        if OllamaLLM is None:
            raise RuntimeError(
                "Transformer LLM unavailable and OllamaLLM is not importable. "
                "Install required dependencies or configure a supported backend."
            )
        llm = OllamaLLM(model=model, ollama_url=ollama_url)

    rag_service = RagService(llm)

    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    @app.get('/health')
    def health():
        return jsonify({"status": "ok"})

    @app.get('/health/llm')
    def health_llm():
        available = llm.alive()
        return jsonify({
            "llm_available": available,
            "model": llm.model,
        }), (200 if available else 503)

    @app.route('/query', methods=['GET'])
    def query_get():
        question = (request.args.get('question') or '').strip()
        if not question:
            return jsonify({
                "status": "ok",
                "message": "Use POST /query with JSON {question}. Example: {\"question\": \"Explain rag_service.py\"}",
            })

        if retrieval_service is None:
            return jsonify({"error": "Retrieval service is not configured."}), 400

        retrieved_chunks = retrieval_service.retrieve(question, top_k=5)
        rag_answer = rag_service.answer_question(question, retrieved_chunks)
        return jsonify({
            "question": rag_answer.question,
            "answer": rag_answer.answer,
            "source_chunks": [chunk.to_dict() for chunk in rag_answer.source_chunks],
            "model_used": rag_answer.model_used,
        })

    @app.route('/query', methods=['POST'])
    def query():
        data = request.get_json(silent=True) or {}
        question = (data.get('question') or '').strip()
        if not question:
            return jsonify({"error": "Field 'question' is required."}), 400

        if retrieval_service is not None:
            try:
                results = retrieval_service.retrieve(question, top_k=int(data.get("top_k", 5)))
                print("[DEBUG] retrieval_service returned:", type(results), "len=", len(results))
                for i, item in enumerate(results[:5], start=1):
                    print(f"[DEBUG] result {i}: {repr(item)}")
            except Exception as e:
                print(f"[ERROR] Retrieval failed: {e!r}")
                traceback.print_exc()
                return jsonify({"error": "Retrieval failed.", "detail": str(e)}), 500
        else:
            results = data.get("retrieval_results", [])
            if not isinstance(results, list):
                return jsonify({"error": "Field 'retrieval_results' must be a list."}), 400

        rag_answer = rag_service.answer_question(question, results)
        return jsonify({
            "question": rag_answer.question,
            "answer": rag_answer.answer,
            "source_chunks": [chunk.to_dict() for chunk in rag_answer.source_chunks],
            "model_used": rag_answer.model_used,
        })

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Route not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error.", "detail": str(e)}), 500

    return app



#----------------------------------------------------------#
#main
#----------------------------------------------------------#
if __name__ == '__main__':
    retrieval_service = None
    try:
        from codebase_rag.retrieval.retrieval_service import RetrievalService

        retrieval_config = SimpleNamespace(
            embedding_model_name=os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
            embedding_batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", 64)),
            device=os.environ.get("DEVICE", "cpu"),
        )
        retrieval_service = RetrievalService.from_settings(retrieval_config)
        print("[INFO] RetrievalService loaded — integrated mode.")
    except Exception as e:
        print(f"[INFO] RetrievalService not available ({e}) — standalone mode.")

    app = create_app(
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        model=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        retrieval_service=retrieval_service,
    )
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
