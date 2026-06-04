export function AskPanel({ question, setQuestion, onSubmit, answer, loading }) {
  return (
    <aside className="panel panel-ask">
      <div className="panel-head">
        <div>
          <h2>Hỏi trợ lý</h2>
          <p>Nhập câu hỏi để nhận gợi ý nhanh về test case hoặc cấu trúc dự án.</p>
        </div>
      </div>
      <form className="ask-form" onSubmit={onSubmit}>
        <label htmlFor="question">Câu hỏi của bạn</label>
        <textarea
          id="question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ví dụ: Hướng dẫn tạo test case cho rag_service.py"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? 'Đang xử lý...' : 'Gửi câu hỏi'}
        </button>
      </form>
      <div className="ask-note">
        <h3>Gợi ý nhanh</h3>
        <p>{answer}</p>
      </div>
    </aside>
  )
}
