export function CodeViewer({ selectedFile }) {
  return (
    <section className="panel panel-code">
      <div className="panel-head">
        <div>
          <span className="section-label">Code Viewer</span>
          <h2>{selectedFile.name}</h2>
        </div>
        <div className="chip">Live preview</div>
      </div>
      <div className="code-window">
        <div className="code-toolbar">
          <span>{selectedFile.path}</span>
          <span>{selectedFile.lang.toUpperCase()}</span>
        </div>
        <pre>
          <code>{selectedFile.content}</code>
        </pre>
      </div>
    </section>
  )
}
