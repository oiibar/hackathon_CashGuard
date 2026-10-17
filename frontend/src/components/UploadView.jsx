import { useState } from 'react'
import Icon from './Icons.jsx'
import UploadDropzone from './UploadDropzone.jsx'
import ZoneEditor from './ZoneEditor.jsx'
import { useT } from '../i18n.js'

export function Stepper({ step }) {
  const t = useT()
  const steps = [t('stepUpload'), t('stepAnalysis'), t('stepReport')]
  return (
    <ol className="stepper">
      {steps.map((label, i) => (
        <li key={label} className={i === step ? 'is-current' : i < step ? 'is-done' : ''}>
          <span className="stepper-dot">{i + 1}</span>
          <span className="stepper-label">{label}</span>
        </li>
      ))}
    </ol>
  )
}

const FEATURES = [
  ['shield', 'featLocalT', 'featLocalD'],
  ['frame', 'featFramesT', 'featFramesD'],
  ['userCheck', 'featHumanT', 'featHumanD'],
]

export default function UploadView({ file, onFile, onStart, apiKey, onApiKey, zoneMode, setZoneMode,
  profile, setProfile, profiles, zones, setZones, videoUrl }) {
  const t = useT()
  const [advanced, setAdvanced] = useState(!apiKey)
  const customOk = Object.values(zones).some((pts) => pts.length >= 3)
  const canStart = file && apiKey && (zoneMode === 'auto' || customOk)

  return (
    <section className="page page-narrow">
      <Stepper step={0} />
      <div className="hero">
        <span className="eyebrow"><Icon name="sparkle" size={14} />{t('badge')}</span>
        <h1>{t('heroTitle')}</h1>
        <p className="lead">{t('heroSub')}</p>
      </div>

      <div className="card upload-card">
        <UploadDropzone file={file} onFile={onFile} />
        <div className="upload-actions">
          <span className="muted small upload-note"><Icon name="lock" size={15} />{t('localNote')}</span>
          <button type="button" className="btn btn-primary" disabled={!canStart} onClick={onStart}>
            <Icon name="arrowRight" size={17} />{t('start')}
          </button>
        </div>
        {!apiKey && file && <p className="text-bad small">{t('needKey')}</p>}

        <details className="advanced" open={advanced} onToggle={(e) => setAdvanced(e.currentTarget.open)}>
          <summary>{t('advanced')}</summary>
          <label className="field">
            <span>{t('apiKey')}</span>
            <input type="password" value={apiKey} placeholder={t('apiKeyPh')} autoComplete="off"
              onChange={(e) => onApiKey(e.target.value.trim())} />
          </label>
          <div className="field">
            <span>{t('zones')}</span>
            <div className="segmented">
              <button type="button" className={zoneMode === 'auto' ? 'is-active' : ''} onClick={() => setZoneMode('auto')}>
                {t('zoneAuto')}
              </button>
              <button type="button" className={zoneMode === 'custom' ? 'is-active' : ''} onClick={() => setZoneMode('custom')}
                disabled={!file}>
                {t('zoneManual')}
              </button>
            </div>
          </div>
          {zoneMode === 'auto' && profiles.length > 1 && (
            <label className="field">
              <span>{t('zoneProfile')}</span>
              <select value={profile} onChange={(e) => setProfile(e.target.value)}>
                {profiles.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </label>
          )}
          {zoneMode === 'custom' && file && <ZoneEditor videoUrl={videoUrl} zones={zones} onChange={setZones} />}
        </details>
      </div>

      <div className="features">
        {FEATURES.map(([icon, title, text]) => (
          <div key={title} className="feature">
            <Icon name={icon} size={20} className="accent" />
            <div><strong>{t(title)}</strong><span>{t(text)}</span></div>
          </div>
        ))}
      </div>
      <p className="muted small center">{t('footNote')}</p>
    </section>
  )
}
