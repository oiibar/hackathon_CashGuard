import Icon from './Icons.jsx'
import { Stepper } from './UploadView.jsx'
import { formatBytes } from './UploadDropzone.jsx'
import { fmtTime } from './Report.jsx'
import { useT } from '../i18n.js'

function stages(phase, uploadPct, progress) {
  if (phase === 'uploading') return [['running', uploadPct], ['next'], ['next'], ['next']]
  if (progress < 0.03) return [['done'], ['running'], ['next'], ['next']]
  if (progress < 0.95) return [['done'], ['done'], ['running'], ['next']]
  return [['done'], ['done'], ['done'], ['running']]
}

export function ProcessingView({ file, videoUrl, duration, phase, uploadPct, job, elapsed, onCancel }) {
  const t = useT()
  const progress = job?.progress || 0
  const pct = phase === 'uploading' ? Math.round(uploadPct * 15) : Math.round(15 + progress * 85)
  const rows = stages(phase, uploadPct, progress)
  const names = ['stUpload', 'stFrames', 'stDetect', 'stReport']
  const queued = job?.stage === 'queued'

  return (
    <section className="page page-narrow">
      <Stepper step={1} />
      <h1 className="page-title">{t('procTitle')}</h1>
      <p className="lead left">{t('procSub')}</p>

      <div className="card proc-card">
        <div className="proc-media">
          <div className="thumb">
            {videoUrl && <video src={videoUrl} muted playsInline preload="metadata" />}
            <span className="tag tag-dark">{file?.name}</span>
            {duration > 0 && <span className="tag tag-dark tag-corner">{fmtTime(duration)}</span>}
          </div>
          <div className="proc-file">
            <strong>{file?.name}</strong>
            <span className="muted small">
              {duration > 0 && `${fmtTime(duration)} · `}{file && formatBytes(file.size)} · {file?.name.split('.').pop().toUpperCase()}
            </span>
            <span className="chip">{t('noFaces')}</span>
          </div>
        </div>
        <div className="proc-status">
          <span className="pill pill-warn"><Icon name="spinner" size={14} className="spin" />
            {queued ? t('queued') : t('inProgress')}</span>
          <div className="proc-head">
            <h2>{t('detecting')}</h2>
            <span className="proc-pct">{pct}%</span>
          </div>
          <div className="bar"><div className="bar-fill" style={{ width: `${pct}%` }} /></div>
          <ul className="checklist">
            {rows.map(([state], i) => (
              <li key={names[i]} className={`is-${state}`}>
                <Icon name={state === 'done' ? 'check' : state === 'running' ? 'spinner' : 'circle'} size={17}
                  className={state === 'running' ? 'spin' : ''} />
                <span>{t(names[i])}</span>
                <em>{t(state === 'done' ? 'done' : state === 'running' ? 'running' : 'next')}</em>
              </li>
            ))}
          </ul>
          <p className="muted small">{t('procHint')} <span className="mono">{fmtTime(elapsed)}</span></p>
        </div>
      </div>

      <div className="note-card">
        <Icon name="shield" size={20} className="accent" />
        <div><strong>{t('privateT')}</strong><span>{t('privateD')}</span></div>
      </div>
      <button type="button" className="btn btn-outline" onClick={onCancel}>{t('cancel')}</button>
    </section>
  )
}

export function ErrorView({ file, error, network, onRetry, onReplace }) {
  const t = useT()
  return (
    <section className="page page-narrow">
      <Stepper step={1} />
      <div className="card error-card">
        <span className="error-icon"><Icon name="fileAlert" size={26} /></span>
        <span className="pill pill-bad">{t('errBadge')}</span>
        <h1>{t('errTitle')}</h1>
        <p className="lead">{network ? t('errNet') : t('errSub')}</p>
        {file && (
          <div className="file-row">
            <Icon name="fileVideo" size={20} className="accent" />
            <div><strong>{file.name}</strong><span className="muted small">{formatBytes(file.size)} · {error}</span></div>
            <Icon name="info" size={18} className="text-bad" />
          </div>
        )}
        <div className="tip-card">
          <strong>{t('beforeT')}</strong>
          <span>{t('beforeD')}</span>
        </div>
        <div className="row-center">
          <button type="button" className="btn btn-primary" onClick={onRetry}><Icon name="refresh" size={16} />{t('retry')}</button>
          <button type="button" className="btn btn-outline" onClick={onReplace}><Icon name="upload" size={16} />{t('replace')}</button>
        </div>
        <p className="muted small">{t('noReport')}</p>
      </div>
    </section>
  )
}
