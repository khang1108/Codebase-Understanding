SYSTEEM_PROMT="""You are an soft ware engineer assistant.
-Your task is anser the question based on codebase using provide sourcecode and your knowledge.
Rules:
-Answer the question based on the provided source code and your knowledge.
-If the question is not related to the provided source code ,say so clearly and do not attempt to answer it.
-Reference file paths when relevant (e.g., "In file 'utils.py', the function 'calculate_sum' is defined as follows: ...").
-Check clearly do note hallucinate code or API not in the context.
-If the question is about a specific function, class, or module, provide a detailed explanation
"""
def build_prompt(question:str,context:list[dict])->str:
    """3/6/2026 Function dùng để xây dựng prompt cho mô hình dựa trên câu hỏi và context đã được xây dựng từ các chunk đã truy xuất."""
    if not context:
        context_str = "No relevant information found in the codebase."
    else:
        parts = []
        for i, chunk in enumerate(context, 1):
            file_path = chunk.get("file_path") or chunk.get("source", "Unknown source")
            content = chunk.get("content", "")
            start_line = chunk.get("start_line", "N/A")
            end_line = chunk.get("end_line", "N/A")
            location_info = f"(File: {file_path}, Lines: {start_line}-{end_line})"
            parts.append(f"Context {i} {location_info}:\n{content}")
        context_str = "\n\n".join(parts)
    prompt=f"""
            {context_str}
            ### user question: {question}
            ### model answer:
            """
    return prompt 