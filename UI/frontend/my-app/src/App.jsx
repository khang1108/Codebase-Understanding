import { useState } from 'react'
import './App.css'
import { FolderExplorer } from './components/FolderExplorer.jsx'
import { CodeViewer } from './components/CodeViewer.jsx'
import { AskPanel } from './components/AskPanel.jsx'
import axiosClient from '../axiosClient.js'
const fileTree = [
  {
    label: 'UI / Frontend',
    children: [
      {
        name: 'App.jsx',
        path: 'UI/frontend/my-app/src/App.jsx',
        lang: 'jsx',
        content: `export default function App() {
  return (
    <div className="app-shell">
      <h1>Giao diện Tool</h1>
    </div>
  )
}`,
      },
      {
        name: 'main.jsx',
        path: 'UI/frontend/my-app/src/main.jsx',
        lang: 'jsx',
        content: `import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)`,
      },
    ],
  },
  {
    label: 'RAG / Backend',
    children: [
      {
        name: 'answer_generator.py',
        path: 'rag/answer_generator.py',
        lang: 'py',
        content: `def generate_answer(context, question):\n    # Sinh câu trả lời dựa trên ngữ cảnh và yêu cầu\n    return 'Trả lời mẫu cho câu hỏi.'`,
      },
      {
        name: 'context_builder.py',
        path: 'rag/context_builder.py',
        lang: 'py',
        content: `class ContextBuilder:\n    def build(self, project_files):\n        return 'Context được tạo từ các file code.'`,
      },
    ],
  },
  {
    label: 'Domain / Port',
    children: [
      {
        name: 'llm_port.py',
        path: 'domain/port/llm_port.py',
        lang: 'py',
        content: `class LLMPort:\n    def request(self, prompt):\n        return 'Yêu cầu LLM đã gửi.'`,
      },
    ],
  },
]

function App() {
  const [selectedFile, setSelectedFile] = useState(fileTree[0].children[0])
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState(
    'Bạn có thể hỏi cách tạo test case, cấu trúc thư mục, hoặc làm quen nhanh với dự án.'
  )
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (!question.trim()) {
      return
    }

    setLoading(true)
    setAnswer('Đang gửi câu hỏi đến backend...')

    try {
      const response = await axiosClient.post('/query', {
        question,
      })
      console.log('query response', response)

      if (response?.data?.answer) {
        setAnswer(response.data.answer)
      } else if (response?.data) {
        setAnswer(JSON.stringify(response.data, null, 2))
      } else {
        setAnswer('Không nhận được câu trả lời từ server.')
      }
    } catch (error) {
      console.error('Query API error:', error)
      setAnswer(
        error?.response?.data?.error ||
          'Lỗi khi gọi API. Vui lòng kiểm tra backend và thử lại.'
      )
    } finally {
      setLoading(false)
      setQuestion('')
    }
  }

  return (
    <div className="app-shell">
      <div className="topbar">
        <div className="brand">
          <div className="brand-mark">C</div>
          <div>
            <p className="brand-label">Codebase Navigator</p>
            <h1>Giao diện khám phá dự án</h1>
          </div>
        </div>
        <div className="status-badge">Dark Galaxy Theme</div>
      </div>

      <div className="workspace">
        <FolderExplorer
          fileTree={fileTree}
          selectedFile={selectedFile}
          onFileSelect={setSelectedFile}
        />
        <CodeViewer selectedFile={selectedFile} />
        <AskPanel
          question={question}
          setQuestion={setQuestion}
          onSubmit={handleSubmit}
          answer={answer}
          loading={loading}
        />
      </div>
    </div>
  )
}

export default App
