import { useEffect, useRef, useState } from 'react'
import Icon from './Icons.jsx'
import { Stepper } from './UploadView.jsx'
import { useT } from '../i18n.js'
import { reviewJob } from '../api.js'

export function fmtTime(sec) {
  const s = Math.max(0, sec || 0)
  const m = Math.floor(s / 60)
  return `${String(m).padStart(2, '0')}:${(s - m * 60).toFixed(1).padStart(4, '0')}`
}

const jpeg = (b64) => `data:image/jpeg;base64,${b64}`

function EvidenceModal({ open, onClose, fileName }) {
  const t = useT()
  const [i, setI] = useState(0)
  useEffect(() => setI(open?.index || 0), [open])
  useEffect(() => {
    if (!open) return
    const onKey = (e) => {
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowRight') setI((x) => Math.min(x + 1, open.v.evidence_frames.length - 1))
      if (e.key === 'ArrowLeft') setI((x) => Math.max(x - 1, 0))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])
  if (!open) return null
  const { v } = open
  const frames = v.evidence_frames
  const f = frames[i]
  return (
    <div className="modal" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="modal-body" onClick={(e) => e.stopPropagation()}>
        <header className="modal-head">
          <div>
            <h2>{t('evidence')}</h2>
            <span className="muted small">{fileName} · {t('frameOf', { i: i + 1, n: frames.length })}</span>
          </div>
          <button type="button" className="icon-btn" onClick={onClose} aria-label={t('close')}><Icon name="x" /></button>
        </header>
        <div className="modal-grid">
          <figure className="modal-frame">
            <img src={jpeg(f.image_base64)} alt={`${v.type} ${fmtTime(f.timestamp_sec)}`} />
            <span className="tag tag-dark tag-corner">{fmtTime(f.timestamp_sec)}</span>
          </figure>
          <aside className="modal-side">
            <span className="pill pill-bad">{t('possible')} · {fmtTime(v.timecode_sec)}</span>
            <h3>{v.type}</h3>
            <h4>{t('whatVisible')}</h4>
            <p>{v.observed || v.description}</p>
            {v.why && <><h4>{t('whyViolation')}</h4><p>{v.why}</p></>}
            <LegalBox basis={v.legal_basis} />
          </aside>
        </div>
        <footer className="modal-nav">
          <button type="button" className="btn btn-outline btn-sm" disabled={i === 0} onClick={() => setI(i - 1)}>
            <Icon name="arrowLeft" size={16} />{t('prev')}
          </button>
          <span className="pill pill-soft">{fmtTime(f.timestamp_sec)} / {fmtTime(v.end_sec ?? f.timestamp_sec)}</span>
          <button type="button" className="btn btn-outline btn-sm" disabled={i === frames.length - 1} onClick={() => setI(i + 1)}>
            {t('nextFrame')}<Icon name="arrowRight" size={16} />
          </button>
        </footer>
        <div className="disclaimer"><Icon name="info" size={16} />{t('disclaimer')}</div>
      </div>
    </div>
  )
}

function LegalBox({ basis }) {
  const t = useT()
  if (!basis?.length) return null
  return (
    <div className="legal-box">
      <span className="legal-label">{t('legalLabel')}</span>
      {basis.map((b, i) => (
        <div key={i} className="legal-item">
          <strong>{b.short}, {b.clause}</strong>
          <q>{b.quote}</q>
          <a href={b.url} target="_blank" rel="noreferrer">{t('openAct')}<Icon name="external" size={13} /></a>
        </div>
      ))}
    </div>
  )
}

function Player({ videoUrl, duration, violations, videoRef }) {
  const [time, setTime] = useState(0)
  const [ok, setOk] = useState(true)
  const seek = (s) => { const el = videoRef.current; if (el) { el.currentTime = s; el.play().catch(() => {}) } }
  return (
    <div className="player">
      {videoUrl && ok ? (
        <video ref={videoRef} src={videoUrl} controls playsInline preload="metadata"
          onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)} onError={() => setOk(false)} />
      ) : (
        <div className="player-empty">
          {violations[0]?.evidence_frames[0] && <img src={jpeg(violations[0].evidence_frames[0].image_base64)} alt="" />}
        </div>
      )}
      <div className="timeline">
        <div className="timeline-track" onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          seek(((e.clientX - r.left) / r.width) * duration)
        }}>
          <div className="timeline-fill" style={{ width: `${Math.min((time / (duration || 1)) * 100, 100)}%` }} />
          {violations.map((v, i) => (
            <button key={i} type="button" className="timeline-mark" title={`${fmtTime(v.timecode_sec)} ${v.type}`}
              style={{ left: `${Math.min((v.timecode_sec / (duration || 1)) * 100, 100)}%` }}
              onClick={(e) => { e.stopPropagation(); seek(v.timecode_sec) }} />
          ))}
        </div>
        <span className="mono timeline-time">{fmtTime(time)} / {fmtTime(duration)}</span>
      </div>
    </div>
  )
}

function MomentCard({ v, id, onSeek, onOpen }) {
  const t = useT()
  return (
    <article className="card moment" id={id}>
      <header className="moment-head">
        <span className="pill pill-bad">{t('possible')}</span>
        <button type="button" className="pill pill-soft pill-btn" onClick={() => onSeek(v.timecode_sec)}>
          <Icon name="clock" size={14} />{fmtTime(v.timecode_sec)}
        </button>
      </header>
      <h3>{v.type}</h3>
      <div className="moment-frames">
        {v.evidence_frames.map((f, i) => (
          <div key={i}>
            <button type="button" className="thumb thumb-btn" onClick={() => onOpen(v, i)}>
              <img src={jpeg(f.image_base64)} alt={`${v.type} ${fmtTime(f.timestamp_sec)}`} loading="lazy" />
              <span className="tag tag-dark tag-corner">{fmtTime(f.timestamp_sec)}</span>
            </button>
            <button type="button" className="link-btn" onClick={() => onOpen(v, i)}>
              <Icon name="expand" size={14} />{t('enlarge')}
            </button>
          </div>
        ))}
      </div>
      <p className="muted small">
        {v.confidence != null && `${t('confidence')} ${Math.round(v.confidence * 100)}%`}
        {v.duration_sec != null && ` · ${t('lasted')} ${v.duration_sec.toFixed(1)} s`}
        {v.source && <> · {t(v.source === 'learned' ? 'srcModel' : 'srcRule')}</>}
        {v.type_id && <> · <code>{v.type_id}</code></>}
      </p>
      <h4>{t('whatVisible')}</h4>
      <p>{v.observed || v.description}</p>
      {v.why && <><h4>{t('whyViolation')}</h4><p>{v.why}</p></>}
      <LegalBox basis={v.legal_basis} />
    </article>
  )
}

function TechDetails({ job }) {
  const t = useT()
  const [json, setJson] = useState(false)
  const meta = job.meta || {}
  const z = meta.zone_detection || {}
  const download = () => {
    const blob = new Blob([JSON.stringify(job, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `report_${job.job_id}.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }
  return (
    <details className="card tech">
      <summary>{t('tech')}</summary>
      <ul className="tech-list">
        <li>{meta.frames_analyzed} {t('frames')} · {meta.sample_fps} {t('fps')} · {t('procTime')} {meta.processing_time_sec} s</li>
        <li>{z.method === 'manual' ? t('zoneManualMeta') : z.localized ? t('zoneFound') : t('zoneNotFound')}</li>
        {meta.timings && (
          <li>{Object.entries(meta.timings).map(([k, v]) => `${k} ${v} s`).join(' · ')}</li>
        )}
        {meta.models && <li>{meta.models.join(', ')}</li>}
        {meta.skipped_rules?.length > 0 && (
          <li>{t('skipped')}: {meta.skipped_rules.map((s) => s.type).join(', ')}</li>
        )}
      </ul>
      <div className="row">
        <button type="button" className="btn btn-outline btn-sm" onClick={download}><Icon name="download" size={15} />{t('download')}</button>
        <button type="button" className="btn btn-outline btn-sm" onClick={() => setJson(!json)}><Icon name="code" size={15} />{t('showJson')}</button>
      </div>
      {json && (
        <pre className="json">
          {JSON.stringify(job, (k, val) => (k === 'image_base64' ? `${val.slice(0, 32)}… (${val.length})` : val), 2)}
        </pre>
      )}
    </details>
  )
}

function ReviewCard({ job, apiKey, types }) {
  const t = useT()
  const first = job.violations?.[0]
  const [label, setLabel] = useState(job.verdict)
  const [time, setTime] = useState(first ? String(first.timecode_sec) : '')
  const [type, setType] = useState(first?.type_id || '')
  const [state, setState] = useState({ saving: false, msg: '', error: '' })
  const save = () => {
    setState({ saving: true, msg: '', error: '' })
    const body = label === 'violation'
      ? { label, timecode_sec: Number(time), type_id: type || null }
      : { label }
    reviewJob(job.job_id, apiKey, body)
      .then((r) => setState({
        saving: false, error: '',
        msg: r.trainable ? t('reviewSaved', { n: r.pending }) + (r.retraining ? ' ' + t('retrainStarted') : '') : t('reviewSavedNoEmb'),
      }))
      .catch((e) => setState({ saving: false, msg: '', error: e.message }))
  }
  const invalid = label === 'violation' && (time === '' || Number.isNaN(Number(time)) || Number(time) < 0)
  return (
    <section className="card review">
      <header className="section-head">
        <h2>{t('reviewTitle')}</h2>
        <span className="muted small">{t('reviewHint')}</span>
      </header>
      <div className="segmented">
        {['violation', 'no_violation'].map((l) => (
          <button key={l} type="button" className={label === l ? 'is-active' : ''} onClick={() => setLabel(l)}>
            {t(l === 'violation' ? 'reviewYes' : 'reviewNo')}
          </button>
        ))}
      </div>
      {label === 'violation' && (
        <div className="review-fields">
          <label>
            <span className="small">{t('reviewTime')}</span>
            <input type="number" min="0" step="0.5" value={time} onChange={(e) => setTime(e.target.value)} />
          </label>
          <label>
            <span className="small">{t('reviewType')}</span>
            <select value={type} onChange={(e) => setType(e.target.value)}>
              <option value="">{t('reviewTypeOther')}</option>
              {(types || []).map((r) => <option key={r.type} value={r.type}>{r.title}</option>)}
            </select>
          </label>
        </div>
      )}
      <div className="row">
        <button type="button" className="btn btn-primary btn-sm" disabled={!apiKey || invalid || state.saving} onClick={save}>
          <Icon name="check" size={15} />{t('reviewSave')}
        </button>
        {state.msg && <span className="small text-good">{state.msg}</span>}
        {state.error && <span className="small text-bad">{state.error}</span>}
      </div>
    </section>
  )
}

export default function Report({ job, videoUrl, fileName, onAnother, apiKey, types }) {
  const t = useT()
  const [modal, setModal] = useState(null)
  const videoRef = useRef(null)
  const meta = job.meta || {}
  const vs = job.violations || []
  const bad = job.verdict === 'violation'
  const duration = meta.duration_sec || 0
  const date = job.created_at ? new Date(job.created_at * 1000) : new Date()

  const seek = (s) => {
    const el = videoRef.current
    if (!el) return
    el.currentTime = s
    el.play().catch(() => {})
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
  const goTo = (i) => document.getElementById(`moment-${i}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })

  return (
    <section className="page">
      <Stepper step={2} />
      <button type="button" className="crumb" onClick={onAnother}><Icon name="arrowLeft" size={15} />{t('crumb')}</button>
      <div className="report-head">
        <div>
          <h1 className="page-title">{t('reportTitle')}</h1>
          <div className="muted small">{fileName || job.filename}</div>
          <div className="chips">
            <span className="chip">{fmtTime(duration)} {t('duration')}</span>
            <span className="chip">{date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}</span>
            <span className="chip">{t('humanReview')}</span>
          </div>
        </div>
        <button type="button" className="btn btn-outline" onClick={onAnother}><Icon name="plus" size={16} />{t('another')}</button>
      </div>

      <div className={`banner ${bad ? 'banner-bad' : 'banner-good'}`}>
        <Icon name={bad ? 'alert' : 'checkCircle'} size={22} />
        <div>
          <strong>{bad ? `${t('badTitle')} (${vs.length})` : t('goodTitle')}</strong>
          <span>{bad ? t('badSub') : t('goodSub')}</span>
        </div>
      </div>
      {meta.stopped_early && (
        <p className="text-bad small">{t('coverageCut')} {Math.round((meta.coverage || 0) * 100)}%</p>
      )}

      <div className="report-grid">
        <Player videoUrl={videoUrl} duration={duration} violations={vs} videoRef={videoRef} />
        <aside className="card card-soft overview">
          <header><h3>{bad ? t('overview') : t('summary')}</h3><Icon name="clipboard" size={18} className="accent" /></header>
          <div className="stats">
            <div><strong className={bad ? '' : 'text-good'}>{vs.length}</strong><span>{t('flagged')}</span></div>
            <div><strong>{fmtTime(duration).replace(/\.\d$/, '')}</strong><span>{t('reviewed')}</span></div>
          </div>
          {bad ? (
            <ul className="overview-list">
              {vs.map((v, i) => (
                <li key={i}>
                  <button type="button" onClick={() => { seek(v.timecode_sec); goTo(i) }}>
                    <span className="pill pill-soft">{fmtTime(v.timecode_sec)}</span>
                    <span>{v.type}</span>
                    <Icon name="arrowUpRight" size={15} className="accent" />
                  </button>
                </li>
              ))}
            </ul>
          ) : <p className="small">{t('cleanNote')}</p>}
          <p className="muted xsmall">{t('overviewNote')}</p>
        </aside>
      </div>

      {bad ? (
        <>
          <div className="section-head">
            <h2>{t('flaggedTitle')}</h2>
            <span className="muted small">{t('flaggedHint')}</span>
          </div>
          <div className="moments">
            {vs.map((v, i) => (
              <MomentCard key={i} id={`moment-${i}`} v={v} onSeek={seek} onOpen={(vv, index) => setModal({ v: vv, index })} />
            ))}
          </div>
        </>
      ) : (
        <div className="card nothing">
          <span className="nothing-icon"><Icon name="checks" size={22} /></span>
          <div><strong>{t('nothingT')}</strong><span className="muted">{t('nothingD')}</span></div>
          <button type="button" className="btn btn-outline" onClick={() => seek(0)}><Icon name="play" size={15} />{t('reviewVideo')}</button>
        </div>
      )}

      <ReviewCard job={job} apiKey={apiKey} types={types} />
      <div className="disclaimer"><Icon name="info" size={16} />{t('disclaimer')}</div>
      <TechDetails job={job} />
      <EvidenceModal open={modal} onClose={() => setModal(null)} fileName={fileName || job.filename} />
    </section>
  )
}
