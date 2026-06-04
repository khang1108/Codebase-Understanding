export function FolderExplorer({ fileTree, selectedFile, onFileSelect }) {
  return (
    <aside className="panel panel-side">
      <div className="panel-head">
        <div>
          <h2>Thư mục</h2>
          <p>Chọn file để xem nhanh cấu trúc và nội dung code.</p>
        </div>
      </div>
      <div className="folders">
        {fileTree.map((section) => (
          <div key={section.label} className="folder-group">
            <h3>{section.label}</h3>
            <ul>
              {section.children.map((file) => (
                <li
                  key={file.path}
                  className={file.path === selectedFile.path ? 'active' : ''}
                  onClick={() => onFileSelect(file)}
                >
                  <span>{file.name}</span>
                  <small>{file.path}</small>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </aside>
  )
}
