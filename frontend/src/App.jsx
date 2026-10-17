import { useEffect, useMemo, useRef, useState } from 'react'
import { analyze, getConfig, getJob, health } from './api.js'
import UploadDropzone from './components/UploadDropzone.jsx'
import ZoneEditor from './components/ZoneEditor.jsx'
import Report, { fmtTime } from './components/Report.jsx'

const KEY_STORAGE = 'cashguard.apiKey'
const LIMIT_SEC = 300

function loadKey() {
  try { return localStorage.getItem(KEY_STORAGE) || '' } catch { return '' }
}
function saveKey(v) {
  try { localStorage.setItem(KEY_STORAGE, v) } catch { /* private mode */ }
}

export default function App() {
  const [apiKey, setApiKey] = useState(loadKey)
  const [server, setServer] = useState(null) // health payload or {error}
  const [config, setConfig] = useState(null)

  const [file, setFile] = useState(null)
  const [zoneMode, setZoneMode] = useState('profile') // profile | custom
  const [profile, setProfile] = useState('')
  const [zones, setZones] = useState({})

  const [phase, setPhase] = useState('idle') // idle | uploading | processing | done | failed
  const [uploadPct, setUploadPct] = useState(0)
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')
  const [startedAt, setStartedAt] = useState(null)
  const [now, setNow] = useState(Date.now())
  const pollRef = useRef(null)

  const videoUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])
  useEffect(() => () => videoUrl && URL.revokeObjectURL(videoUrl), [videoUrl])

  useEffect(() => {
    health().then(setServer).catch((e) => setServer({ error: e.message }))
  }, [])

  useEffect(() => {
    saveKey(apiKey)
    if (!apiKey) { setConfig(null); return }
    getConfig(apiKey)
      .then((c) => { setConfig(c); setProfile((p) => p || c.default_profile) })
      .catch(() => setConfig(null))
  }, [apiKey])

  // Elapsed timer while busy.
  useEffect(() => {
    if (phase !== 'uploading' && phase !== 'processing') return
    const id = setInterval(() => setNow(Date.now()), 250)
    return () => clearInterval(id)
  }, [phase])

  useEffect(() => () => clearTimeout(pollRef.current), [])

  const poll = (jobId) => {
    getJob(jobId, apiKey)
      .then((body) => {
        setJob(body)
        if (body.status === 'processing') {
          pollRef.current = setTimeout(() => poll(jobId), 1500)
        } else {
          setPhase(body.status === 'done' ? 'done' : 'failed')
          if (body.status === 'failed') setError(body.error || 'Обработка завершилась ошибкой')
        }
      })
      .catch((e) => {
        // transient network error: keep polling, the job lives on the server
        setError(e.message)
        pollRef.current = setTimeout(() => poll(jobId), 3000)
      })
  }

  const customZonesValid = Object.values(zones).some((pts) => pts.length >= 3)

  const start = async () => {
    setError('')
    setJob(null)
    setUploadPct(0)
    setStartedAt(Date.now())
    setPhase('uploading')
    try {
      const usable = Object.fromEntries(Object.entries(zones).filter(([, pts]) => pts.length >= 3))
      const { job_id } = await analyze(
        file,
        { apiKey, zones: zoneMode === 'custom' && customZonesValid ? usable : null, profile },
        setUploadPct,
      )
      setJob({ job_id, status: 'processing', progress: 0 })
      setPhase('processing')
      poll(job_id)
    } catch (e) {
      setError(e.message)
      setPhase('failed')
    }
  }

  const reset = () => {
    clearTimeout(pollRef.current)
    setPhase('idle')
    setJob(null)
    setError('')
  }

  const busy = phase === 'uploading' || phase === 'processing'
  const elapsed = startedAt ? (now - startedAt) / 1000 : 0
  const canStart = file && apiKey && !busy && (zoneMode === 'profile' || customZonesValid)

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden />
          <div>
            <div className="brand-name">CashGuard</div>
            <div className="brand-sub">Контроль кассовой дисциплины по видео</div>
          </div>
        </div>
        <div className="topbar-right">
          <span className={`server-dot ${server?.status === 'ok' ? 'ok' : server ? 'bad' : ''}`} />
          <span className="server-text">
            {server?.status === 'ok'
              ? `Сервер готов · ${server.detector_fps?.[0] ?? '?'} к/с`
              : server?.error ? 'Сервер недоступен' : 'Проверка сервера…'}
          </span>
          <label className="key-field">
            <span>X-API-Key</span>
            <input
              type="password"
              value={apiKey}
              placeholder="ключ доступа"
              onChange={(e) => setApiKey(e.target.value.trim())}
              autoComplete="off"
            />
          </label>
        </div>
      </header>

      <main className="layout">
        <section className="panel">
          <h2 className="panel-title"><span className="step">1</span> Видео</h2>
          <UploadDropzone file={file} onFile={(f) => { setFile(f); setZones({}); reset() }} disabled={busy} />

          {file && (
            <>
              <h2 className="panel-title"><span className="step">2</span> Зоны кадра</h2>
              <div className="segmented" role="tablist">
                <button type="button" className={zoneMode === 'profile' ? 'is-active' : ''} onClick={() => setZoneMode('profile')} disabled={busy}>
                  Готовый профиль
                </button>
                <button type="button" className={zoneMode === 'custom' ? 'is-active' : ''} onClick={() => setZoneMode('custom')} disabled={busy}>
                  Разметить вручную
                </button>
              </div>
              {zoneMode === 'profile' ? (
                <label className="select-field">
                  <span>Профиль камеры</span>
                  <select value={profile} onChange={(e) => setProfile(e.target.value)} disabled={busy || !config}>
                    {Object.keys(config?.zone_profiles || { default: 1 }).map((p) => <option key={p} value={p}>{p}</option>)}
                  </select>
                  <small className="muted">default: весь кадр — кассовая зона. Так обрабатываются запросы жюри без зон.</small>
                </label>
              ) : (
                <ZoneEditor videoUrl={videoUrl} zones={zones} onChange={setZones} />
              )}

              <h2 className="panel-title"><span className="step">3</span> Анализ</h2>
              {!apiKey && <p className="field-error">Введите X-API-Key в шапке страницы.</p>}
              <div className="run-row">
                <button type="button" className="btn-primary" disabled={!canStart} onClick={start}>
                  {busy ? 'Идёт обработка…' : 'Запустить анализ'}
                </button>
                {(phase === 'done' || phase === 'failed') && (
                  <button type="button" className="btn-ghost" onClick={reset}>Сбросить</button>
                )}
              </div>
            </>
          )}

          {phase !== 'idle' && (
            <div className={`status status-${phase}`}>
              <div className="status-row">
                <span className="status-label">
                  {phase === 'uploading' && 'Загрузка файла'}
                  {phase === 'processing' && (job?.stage === 'queued' ? 'В очереди' : 'Обработка')}
                  {phase === 'done' && 'Готово'}
                  {phase === 'failed' && 'Ошибка'}
                </span>
                <span className={`elapsed${elapsed > LIMIT_SEC ? ' over' : ''}`}>
                  {fmtTime(elapsed)} / 05:00
                </span>
              </div>
              {busy && (
                <div className="progress">
                  <div
                    className="progress-bar"
                    style={{ width: `${Math.round((phase === 'uploading' ? uploadPct : job?.progress || 0) * 100)}%` }}
                  />
                </div>
              )}
              {job?.job_id && <code className="job-id">job_id: {job.job_id}</code>}
              {error && <p className="field-error">{error}</p>}
            </div>
          )}
        </section>

        <section className="panel panel-wide">
          {phase === 'done' && job ? (
            <Report job={job} videoUrl={videoUrl} />
          ) : (
            <div className="placeholder">
              <h2>Отчёт появится здесь</h2>
              <p className="muted">
                Сервис проверяет видео на нарушения кассовой дисциплины: телефон и личные вещи
                в кассовой зоне, посторонние лица, кассир покинул рабочее место, перекрытая
                камера. Каждое нарушение — с таймкодом, кадрами и ссылкой на акт НБРК на adilet.zan.kz.
              </p>
              <p className="muted small">Обработка локальная: видео не покидает сервер, распознавание лиц не используется.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
