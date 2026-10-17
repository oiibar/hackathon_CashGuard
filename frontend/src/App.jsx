import { useEffect, useMemo, useRef, useState } from 'react'
import { analyze, getConfig, getJob, health } from './api.js'
import { I18n, makeT } from './i18n.js'
import Icon from './components/Icons.jsx'
import UploadView from './components/UploadView.jsx'
import { ErrorView, ProcessingView } from './components/StatusViews.jsx'
import Report from './components/Report.jsx'
import { HowView, TypesView } from './components/InfoViews.jsx'
import ModelView from './components/ModelView.jsx'

const store = {
  get: (k) => { try { return localStorage.getItem(k) || '' } catch { return '' } },
  set: (k, v) => { try { localStorage.setItem(k, v) } catch { return undefined } },
}

function initialKey() {
  return new URLSearchParams(window.location.search).get('key') || store.get('cashguard.apiKey')
}

function initialTheme() {
  const saved = store.get('cashguard.theme')
  if (saved) return saved
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function useVideoDuration(url) {
  const [d, setD] = useState(0)
  useEffect(() => {
    setD(0)
    if (!url) return
    const v = document.createElement('video')
    v.preload = 'metadata'
    v.onloadedmetadata = () => setD(Number.isFinite(v.duration) ? v.duration : 0)
    v.src = url
    return () => { v.src = '' }
  }, [url])
  return d
}

export default function App() {
  const [lang, setLang] = useState(() => store.get('cashguard.lang') || 'ru')
  const t = useMemo(() => makeT(lang), [lang])
  const [theme, setTheme] = useState(initialTheme)
  const [view, setView] = useState('analyze')
  const [menu, setMenu] = useState(false)

  const [apiKey, setApiKey] = useState(initialKey)
  const [server, setServer] = useState(null)
  const [config, setConfig] = useState(null)

  const [file, setFile] = useState(null)
  const [zoneMode, setZoneMode] = useState('auto')
  const [profile, setProfile] = useState('auto')
  const [zones, setZones] = useState({})

  const [phase, setPhase] = useState('idle')
  const [uploadPct, setUploadPct] = useState(0)
  const [job, setJob] = useState(null)
  const [error, setError] = useState({ text: '', network: false })
  const [startedAt, setStartedAt] = useState(0)
  const [now, setNow] = useState(Date.now())
  const pollRef = useRef(null)
  const runRef = useRef(0)

  const videoUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])
  useEffect(() => () => videoUrl && URL.revokeObjectURL(videoUrl), [videoUrl])
  const duration = useVideoDuration(videoUrl)

  useEffect(() => { store.set('cashguard.lang', lang); document.documentElement.lang = lang === 'kz' ? 'kk' : lang }, [lang])
  useEffect(() => { store.set('cashguard.apiKey', apiKey) }, [apiKey])
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    store.set('cashguard.theme', theme)
  }, [theme])
  useEffect(() => {
    health().then(setServer).catch(() => setServer({ error: true }))
    getConfig().then((c) => { setConfig(c); setProfile(c.default_profile) }).catch(() => {})
  }, [])

  useEffect(() => {
    if (phase !== 'uploading' && phase !== 'processing') return
    const id = setInterval(() => setNow(Date.now()), 500)
    return () => clearInterval(id)
  }, [phase])
  useEffect(() => () => clearTimeout(pollRef.current), [])

  const poll = (jobId, run) => {
    getJob(jobId, apiKey)
      .then((body) => {
        if (run !== runRef.current) return
        setJob(body)
        if (body.status === 'processing') pollRef.current = setTimeout(() => poll(jobId, run), 1500)
        else if (body.status === 'done') setPhase('done')
        else { setError({ text: body.error || '', network: false }); setPhase('failed') }
      })
      .catch(() => {
        if (run === runRef.current) pollRef.current = setTimeout(() => poll(jobId, run), 3000)
      })
  }

  const start = async () => {
    const run = ++runRef.current
    setError({ text: '', network: false })
    setJob(null)
    setUploadPct(0)
    setStartedAt(Date.now())
    setPhase('uploading')
    window.scrollTo({ top: 0 })
    try {
      const custom = Object.fromEntries(Object.entries(zones).filter(([, pts]) => pts.length >= 3))
      const { job_id } = await analyze(file, {
        apiKey,
        zones: zoneMode === 'custom' && Object.keys(custom).length ? custom : null,
        profile: zoneMode === 'auto' ? profile : null,
      }, setUploadPct)
      if (run !== runRef.current) return
      setJob({ job_id, status: 'processing', progress: 0 })
      setPhase('processing')
      poll(job_id, run)
    } catch (e) {
      if (run !== runRef.current) return
      setError({ text: e.message, network: true })
      setPhase('failed')
    }
  }

  const reset = () => {
    runRef.current++
    clearTimeout(pollRef.current)
    setPhase('idle')
    setJob(null)
    setView('analyze')
    window.scrollTo({ top: 0 })
  }
  const another = () => { reset(); setFile(null); setZones({}) }

  const go = (v) => { setView(v); setMenu(false); window.scrollTo({ top: 0 }) }
  const elapsed = startedAt ? (now - startedAt) / 1000 : 0

  let body
  if (view === 'how') body = <HowView />
  else if (view === 'types') body = <TypesView rules={config?.rules} />
  else if (view === 'model') body = <ModelView apiKey={apiKey} />
  else if (phase === 'uploading' || phase === 'processing') {
    body = <ProcessingView file={file} videoUrl={videoUrl} duration={duration} phase={phase} uploadPct={uploadPct}
      job={job} elapsed={elapsed} onCancel={reset} />
  } else if (phase === 'done' && job) {
    body = <Report job={job} videoUrl={videoUrl} fileName={file?.name} onAnother={another} apiKey={apiKey}
      types={config?.rules} />
  } else if (phase === 'failed') {
    body = <ErrorView file={file} error={error.text} network={error.network} onRetry={start} onReplace={another} />
  } else {
    body = <UploadView file={file} onFile={(f) => { setFile(f); setZones({}) }} onStart={start}
      apiKey={apiKey} onApiKey={setApiKey} zoneMode={zoneMode} setZoneMode={setZoneMode}
      profile={profile} setProfile={setProfile} profiles={Object.keys(config?.zone_profiles || {})}
      zones={zones} setZones={setZones} videoUrl={videoUrl} />
  }

  const nav = [['analyze', 'navAnalyze'], ['how', 'navHow'], ['types', 'navTypes'], ['model', 'navModel']]
  return (
    <I18n.Provider value={{ lang, t }}>
      <div className="app">
        <header className="topbar">
          <button type="button" className="brand" onClick={() => go('analyze')}>
            <span className="logo">forte<span>.</span></span>
            <span className="brand-sep" />
            <span className="brand-name">CashGuard</span>
          </button>
          <nav className={`nav${menu ? ' is-open' : ''}`}>
            {nav.map(([v, key]) => (
              <button key={v} type="button" className={view === v ? 'is-active' : ''} onClick={() => go(v)}>{t(key)}</button>
            ))}
            <div className="langs">
              {['ru', 'kz', 'en'].map((l) => (
                <button key={l} type="button" className={lang === l ? 'is-active' : ''} onClick={() => setLang(l)}>
                  {l.toUpperCase()}
                </button>
              ))}
            </div>
          </nav>
          {server?.error && <span className="server-down" title={t('serverDown')} />}
          <button type="button" className="icon-btn theme-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-label={t(theme === 'dark' ? 'themeLight' : 'themeDark')} title={t(theme === 'dark' ? 'themeLight' : 'themeDark')}>
            <Icon name={theme === 'dark' ? 'sun' : 'moon'} />
          </button>
          <button type="button" className="icon-btn menu-btn" onClick={() => setMenu(!menu)} aria-label="menu">
            <Icon name="menu" />
          </button>
        </header>

        <main>{body}</main>

        <footer className="footer">
          <span>{t('footer')}</span>
          <span>{t('footerRight')}</span>
        </footer>
      </div>
    </I18n.Provider>
  )
}
