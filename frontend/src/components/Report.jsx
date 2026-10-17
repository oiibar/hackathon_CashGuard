import { useState } from 'react'

export function fmtTime(sec) {
  const s = Math.max(0, sec || 0)
  const m = Math.floor(s / 60)
  return `${String(m).padStart(2, '0')}:${(s - m * 60).toFixed(1).padStart(4, '0')}`
}

const SEVERITY = { high: 'Высокая', medium: 'Средняя', low: 'Низкая' }

function FrameModal({ frame, title, onClose }) {
  if (!frame) return null
  return (
    <div className="modal" role="dialog" aria-modal="true" onClick={onClose}>
      <figure className="modal-body" onClick={(e) => e.stopPropagation()}>
        <img src={`data:image/jpeg;base64,${frame.image_base64}`} alt={`Кадр ${fmtTime(frame.timecode_sec)}`} />
        <figcaption>
          <span>{title} · {fmtTime(frame.timecode_sec)}</span>
          <button type="button" className="btn-ghost" onClick={onClose}>Закрыть</button>
        </figcaption>
      </figure>
    </div>
  )
}

function ViolationCard({ v, index, onSeek, onOpen }) {
  return (
    <article className={`violation sev-${v.severity || 'medium'}`}>
      <header className="violation-head">
        <span className="violation-num">{String(index + 1).padStart(2, '0')}</span>
        <div className="violation-titles">
          <h3>{v.title || v.type}</h3>
          <code>{v.type}</code>
        </div>
        <button type="button" className="timecode" onClick={() => onSeek(v.timecode_sec)} title="Перейти к моменту в видео">
          {fmtTime(v.timecode_sec)}
          {v.end_sec != null && <span className="timecode-end">→ {fmtTime(v.end_sec)}</span>}
        </button>
      </header>

      <dl className="violation-facts">
        {v.severity && <div><dt>Критичность</dt><dd>{SEVERITY[v.severity] || v.severity}</dd></div>}
        {v.duration_sec != null && <div><dt>Длительность</dt><dd>{v.duration_sec.toFixed(1)} с</dd></div>}
        {v.confidence != null && <div><dt>Уверенность модели</dt><dd>{Math.round(v.confidence * 100)}%</dd></div>}
      </dl>

      <p className="violation-desc">{v.description}</p>

      {v.legal_basis?.length > 0 && (
        <ul className="legal">
          {v.legal_basis.map((b, i) => (
            <li key={i}>
              <a href={b.url} target="_blank" rel="noreferrer">{b.short} — {b.clause}</a>
              <span className="muted"> {b.summary}</span>
            </li>
          ))}
        </ul>
      )}

      <div className="frames">
        {v.evidence_frames.map((f, i) => (
          <button key={i} type="button" className="frame" onClick={() => onOpen(f, v.title || v.type)}>
            <img src={`data:image/jpeg;base64,${f.image_base64}`} alt={`Кадр-доказательство ${fmtTime(f.timecode_sec)}`} loading="lazy" />
            <span className="frame-tc">{fmtTime(f.timecode_sec)}</span>
          </button>
        ))}
      </div>
    </article>
  )
}

export default function Report({ job, videoUrl }) {
  const [modal, setModal] = useState(null)
  const [showJson, setShowJson] = useState(false)
  const [videoOk, setVideoOk] = useState(true)
  const isViolation = job.verdict === 'violation'
  const meta = job.meta || {}

  const seek = (t) => {
    const el = document.getElementById('report-video')
    if (!el) return
    el.currentTime = t
    el.play().catch(() => {})
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  const downloadJson = () => {
    const blob = new Blob([JSON.stringify(job, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `report_${job.job_id}.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <section className="report">
      <div className={`verdict ${isViolation ? 'is-bad' : 'is-good'}`}>
        <div>
          <div className="verdict-label">Вердикт</div>
          <div className="verdict-value">{isViolation ? 'Выявлены нарушения' : 'Нарушений не выявлено'}</div>
          <code className="verdict-code">verdict: {job.verdict}</code>
        </div>
        <div className="verdict-count">
          <span>{job.violations.length}</span>
          <small>нарушени{job.violations.length === 1 ? 'е' : 'й'}</small>
        </div>
      </div>

      <dl className="meta-grid">
        <div><dt>Длительность видео</dt><dd>{fmtTime(meta.duration_sec)}</dd></div>
        <div><dt>Проанализировано кадров</dt><dd>{meta.frames_analyzed ?? '—'} <span className="muted">({meta.sample_fps} к/с)</span></dd></div>
        <div><dt>Покрытие видео</dt><dd>{meta.coverage != null ? `${Math.round(meta.coverage * 100)}%` : '—'}</dd></div>
        <div><dt>Время обработки</dt><dd>{meta.processing_time_sec != null ? `${meta.processing_time_sec} с` : '—'}</dd></div>
      </dl>

      {videoUrl && videoOk && (
        <video id="report-video" className="report-video" src={videoUrl} controls playsInline preload="metadata"
          onError={() => setVideoOk(false)} />
      )}
      {videoUrl && !videoOk && (
        <p className="muted small">Браузер не может воспроизвести этот файл. Таймкоды и кадры ниже — из анализа на сервере.</p>
      )}

      {job.violations.map((v, i) => (
        <ViolationCard key={i} v={v} index={i} onSeek={seek} onOpen={(frame, title) => setModal({ frame, title })} />
      ))}

      {meta.skipped_rules?.length > 0 && (
        <details className="skipped">
          <summary>Не проверялось: {meta.skipped_rules.length} правил(а)</summary>
          <ul>
            {meta.skipped_rules.map((s) => <li key={s.type}><code>{s.type}</code> — {s.reason}</li>)}
          </ul>
        </details>
      )}

      <div className="report-actions">
        <button type="button" className="btn-secondary" onClick={downloadJson}>Скачать JSON</button>
        <button type="button" className="btn-ghost" onClick={() => setShowJson((s) => !s)}>
          {showJson ? 'Скрыть ответ API' : 'Показать ответ API'}
        </button>
      </div>
      {showJson && (
        <pre className="json">
          {JSON.stringify(job, (k, val) => (k === 'image_base64' ? `${val.slice(0, 40)}… (${val.length} симв.)` : val), 2)}
        </pre>
      )}

      <FrameModal frame={modal?.frame} title={modal?.title} onClose={() => setModal(null)} />
    </section>
  )
}
