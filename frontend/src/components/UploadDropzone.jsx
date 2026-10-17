import { useRef, useState } from 'react'
import Icon from './Icons.jsx'
import { useT } from '../i18n.js'

const EXT = ['.mov', '.mp4', '.m4v', '.avi', '.mkv']
const ACCEPT = EXT.join(',') + ',video/quicktime,video/mp4'

export function formatBytes(n) {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  if (n < 1024 ** 3) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}

export default function UploadDropzone({ file, onFile, disabled }) {
  const t = useT()
  const inputRef = useRef(null)
  const [over, setOver] = useState(false)
  const [error, setError] = useState('')

  const pick = (f) => {
    if (!f) return
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
    if (!EXT.includes(ext)) { setError(t('dropWrongType')); return }
    setError('')
    onFile(f)
  }
  const open = () => !disabled && inputRef.current?.click()

  return (
    <div
      className={`dropzone${over ? ' is-over' : ''}${file ? ' has-file' : ''}`}
      role="button"
      tabIndex={0}
      onClick={open}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && open()}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); if (!disabled) pick(e.dataTransfer.files?.[0]) }}
    >
      <input ref={inputRef} type="file" accept={ACCEPT} hidden
        onChange={(e) => { pick(e.target.files?.[0]); e.target.value = '' }} />
      <span className="dz-icon"><Icon name={file ? 'fileVideo' : 'upload'} size={22} /></span>
      {file ? (
        <>
          <div className="dz-title">{file.name}</div>
          <div className="dz-sub">{formatBytes(file.size)}</div>
          <span className="btn btn-outline btn-sm"><Icon name="folder" size={16} />{t('replaceFile')}</span>
        </>
      ) : (
        <>
          <div className="dz-title">
            <span className="only-wide">{t('dropTitle')}</span>
            <span className="only-narrow">{t('dropTitleMobile')}</span>
          </div>
          <div className="dz-sub">{t('dropSub')}</div>
          <span className="btn btn-outline btn-sm"><Icon name="folder" size={16} />{t('browse')}</span>
        </>
      )}
      <div className="dz-hint">{error ? <span className="text-bad">{error}</span> : t('dropHint')}</div>
    </div>
  )
}
