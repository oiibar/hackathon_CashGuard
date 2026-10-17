import { useRef, useState } from 'react'

const ACCEPT = '.mov,.mp4,.m4v,.avi,.mkv,video/quicktime,video/mp4'

export function formatBytes(n) {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} КБ`
  if (n < 1024 ** 3) return `${(n / 1024 / 1024).toFixed(1)} МБ`
  return `${(n / 1024 ** 3).toFixed(2)} ГБ`
}

// Native drag-and-drop, no library. Click opens the file picker too.
export default function UploadDropzone({ file, onFile, disabled }) {
  const inputRef = useRef(null)
  const [over, setOver] = useState(false)
  const [error, setError] = useState('')

  const pick = (f) => {
    if (!f) return
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
    if (!ACCEPT.split(',').includes(ext)) {
      setError(`Формат ${ext || '?'} не поддерживается. Нужен .mov (или .mp4)`)
      return
    }
    setError('')
    onFile(f)
  }

  return (
    <div>
      <div
        className={`dropzone${over ? ' is-over' : ''}${disabled ? ' is-disabled' : ''}${file ? ' has-file' : ''}`}
        role="button"
        tabIndex={0}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !disabled && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          if (!disabled) pick(e.dataTransfer.files?.[0])
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => { pick(e.target.files?.[0]); e.target.value = '' }}
        />
        {file ? (
          <div className="dz-file">
            <span className="dz-icon" aria-hidden>▶</span>
            <div>
              <div className="dz-name">{file.name}</div>
              <div className="dz-sub">{formatBytes(file.size)} · нажмите, чтобы выбрать другой файл</div>
            </div>
          </div>
        ) : (
          <div className="dz-empty">
            <div className="dz-big">Перетащите видео сюда</div>
            <div className="dz-sub">или нажмите, чтобы выбрать файл · .mov, .mp4</div>
          </div>
        )}
      </div>
      {error && <p className="field-error">{error}</p>}
    </div>
  )
}
