import { useCallback, useEffect, useState } from 'react'
import Icon from './Icons.jsx'
import { activateVersion, getModel, retrain } from '../api.js'
import { useT } from '../i18n.js'

const pct = (x) => (x == null ? '—' : `${Math.round(x * 100)}%`)
const num = (x, d = 2) => (x == null ? '—' : Number(x).toFixed(d))
const sec = (x) => (x == null ? '—' : `${Number(x).toFixed(1)} s`)

function Tile({ label, value, hint }) {
  return (
    <div className="tile">
      <span className="tile-label">{label}</span>
      <strong className="tile-value">{value}</strong>
      {hint && <span className="tile-hint">{hint}</span>}
    </div>
  )
}

function Confusion({ m }) {
  const t = useT()
  const cell = (n, good, label) => (
    <div className={`cm-cell ${good ? 'cm-good' : 'cm-bad'}`}>
      <strong>{n}</strong><span>{label}</span>
    </div>
  )
  return (
    <div className="cm">
      <span />
      <span className="cm-head">{t('predViolation')}</span>
      <span className="cm-head">{t('predClean')}</span>
      <span className="cm-side">{t('actViolation')}</span>
      {cell(m.tp, true, 'TP')}
      {cell(m.fn, false, 'FN')}
      <span className="cm-side">{t('actClean')}</span>
      {cell(m.fp, false, 'FP')}
      {cell(m.tn, true, 'TN')}
    </div>
  )
}

function RocChart({ points, auc }) {
  const t = useT()
  const [hover, setHover] = useState(null)
  const W = 300
  const P = 34
  const x = (v) => P + v * (W - P - 10)
  const y = (v) => W - P - v * (W - P - 10)
  const d = points.map(([f, r], i) => `${i ? 'L' : 'M'}${x(f).toFixed(1)},${y(r).toFixed(1)}`).join('')
  return (
    <figure className="roc">
      <svg viewBox={`0 0 ${W} ${W}`} role="img" aria-label={`ROC, AUC ${num(auc)}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g} className="roc-grid">
            <line x1={x(0)} x2={x(1)} y1={y(g)} y2={y(g)} />
            <text x={x(0) - 6} y={y(g) + 4} textAnchor="end">{g}</text>
            <text x={x(g)} y={y(0) + 16} textAnchor="middle">{g}</text>
          </g>
        ))}
        <line className="roc-chance" x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} />
        <path className="roc-line" d={d} />
        {points.map(([f, r], i) => (
          <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
            <circle className="roc-hit" cx={x(f)} cy={y(r)} r={10} />
            <circle className={`roc-dot${hover === i ? ' is-on' : ''}`} cx={x(f)} cy={y(r)} r={hover === i ? 5 : 3} />
          </g>
        ))}
        {hover != null && (
          <text className="roc-tip" x={x(0.98)} y={y(0.06)} textAnchor="end">
            FPR {num(points[hover][0])} · TPR {num(points[hover][1])}
          </text>
        )}
      </svg>
      <figcaption className="muted xsmall">{t('rocCaption')} · AUC {num(auc)}</figcaption>
    </figure>
  )
}

function SystemTable({ system }) {
  const t = useT()
  const rows = [['always', t('sysAlways')], ['rules', t('sysRules')], ['model', t('sysModel')], ['system', t('sysAll')]]
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>{t('variant')}</th><th>{t('accuracy')}</th><th>Recall</th><th>{t('falseAlarms')}</th>
            <th>Precision</th><th>F1</th><th>MCC</th><th>{t('tcMedian')}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, label]) => {
            const s = system[key]
            if (!s) return null
            return (
              <tr key={key} className={key === 'system' ? 'is-strong' : ''}>
                <td>{label}</td>
                <td>{s.tp + s.tn}/{s.videos}</td>
                <td>{s.tp}/{s.positives}</td>
                <td>{s.fp}/{s.negatives}</td>
                <td>{pct(s.precision)}</td>
                <td>{num(s.f1)}</td>
                <td>{num(s.mcc)}</td>
                <td>{sec(s.timecode?.median_sec)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function FitCard({ fit, search, params }) {
  const t = useT()
  const tone = fit.state === 'good' ? 'pill-good' : fit.state === 'unknown' ? 'pill-soft' : 'pill-warn'
  return (
    <section className="card fit">
      <header className="section-head">
        <h2>{t('fitTitle')}</h2>
        <span className={`pill ${tone}`}>{t(`fit_${fit.state}`)}</span>
      </header>
      <div className="tiles tiles-3">
        <Tile label={t('trainAuc')} value={num(fit.train_moment_roc_auc)} />
        <Tile label={t('cvAuc')} value={num(fit.cv_moment_roc_auc)} />
        <Tile label={t('gap')} value={num(fit.gap)} hint={t('gapHint')} />
      </div>
      <p className="muted small">{t('fitExplain', { k: params.k, c: params.c, folds: params.folds })}</p>
      <div className="table-wrap">
        <table className="table table-compact">
          <thead><tr><th>PCA k</th><th>C</th><th>MCC</th><th>F1</th><th>ROC-AUC</th><th>TP</th><th>FP</th><th>FN</th></tr></thead>
          <tbody>
            {search.map((r) => (
              <tr key={`${r.k}-${r.c}`} className={r.selected ? 'is-strong' : ''}>
                <td>{r.k}</td><td>{r.c}</td><td>{num(r.mcc)}</td><td>{num(r.f1)}</td><td>{num(r.roc_auc)}</td>
                <td>{r.tp}</td><td>{r.fp}</td><td>{r.fn}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function LearningCard({ learning, apiKey, onChanged }) {
  const t = useT()
  const [msg, setMsg] = useState('')
  const start = () => {
    setMsg('')
    retrain(apiKey).then(() => { setMsg(t('retrainStarted')); onChanged() }).catch((e) => setMsg(e.message))
  }
  const last = learning.last
  return (
    <section className="card">
      <header className="section-head">
        <h2>{t('learnTitle')}</h2>
        <span className={`pill ${learning.enabled ? 'pill-good' : 'pill-soft'}`}>{learning.enabled ? t('learnOn') : t('learnOff')}</span>
      </header>
      <p className="muted small">{t('learnExplain', { n: learning.min_reviews })}</p>
      <div className="tiles tiles-3">
        <Tile label={t('reviewsTotal')} value={learning.reviews} />
        <Tile label={t('reviewsPending')} value={learning.pending} hint={learning.auto_retrain ? t('autoAt', { n: learning.min_reviews }) : t('autoOff')} />
        <Tile label={t('cvFolds')} value={learning.cv_folds || 'LOVO'} />
      </div>
      {last && <p className="small">{last.error ? `${t('lastError')}: ${last.error}` : `v${last.version}: ${last.accepted ? t('promoted') : t('rejected')} — ${last.decision}`}</p>}
      <div className="row">
        <button type="button" className="btn btn-primary btn-sm" disabled={!apiKey || learning.running || !learning.enabled} onClick={start}>
          <Icon name={learning.running ? 'spinner' : 'refresh'} size={15} className={learning.running ? 'spin' : ''} />
          {learning.running ? t('training') : t('retrainNow')}
        </button>
        {!apiKey && <span className="muted xsmall">{t('needKey')}</span>}
        {msg && <span className="small">{msg}</span>}
      </div>
    </section>
  )
}

function Versions({ versions, active, apiKey, onChanged }) {
  const t = useT()
  const [err, setErr] = useState('')
  const use = (id) => activateVersion(id, apiKey).then(onChanged).catch((e) => setErr(e.message))
  return (
    <section className="card">
      <header className="section-head"><h2>{t('versionsTitle')}</h2></header>
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr><th>{t('version')}</th><th>{t('date')}</th><th>{t('status')}</th><th>{t('videos')}</th><th>Recall</th>
              <th>{t('falseAlarms')}</th><th>MCC</th><th>{t('decision')}</th><th /></tr>
          </thead>
          <tbody>
            {versions.map((v) => (
              <tr key={v.id} className={v.id === active ? 'is-strong' : ''}>
                <td>v{v.id}</td>
                <td>{v.created_at ? new Date(v.created_at * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' }) : t('baseModel')}</td>
                <td><span className={`pill ${v.id === active ? 'pill-good' : v.status === 'rejected' ? 'pill-bad' : 'pill-soft'}`}>{t(`st_${v.id === active ? 'active' : v.status}`)}</span></td>
                <td>{v.params?.videos ?? v.metrics?.videos ?? '—'}{v.params?.reviewed ? ` (+${v.params.reviewed})` : ''}</td>
                <td>{v.metrics?.tp != null ? `${v.metrics.tp}/${v.metrics.tp + v.metrics.fn}` : '—'}</td>
                <td>{v.metrics?.fp != null ? `${v.metrics.fp}/${v.metrics.fp + v.metrics.tn}` : '—'}</td>
                <td>{num(v.metrics?.mcc)}</td>
                <td className="xsmall muted">{v.decision}</td>
                <td>
                  {v.id !== active && (
                    <button type="button" className="btn btn-outline btn-sm" disabled={!apiKey} onClick={() => use(v.id)}>{t('activate')}</button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {err && <p className="text-bad small">{err}</p>}
    </section>
  )
}

export default function ModelView({ apiKey }) {
  const t = useT()
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const load = useCallback(() => { getModel().then(setData).catch((e) => setError(e.message)) }, [])
  useEffect(load, [load])
  useEffect(() => {
    if (!data?.learning?.running) return
    const id = setTimeout(load, 3000)
    return () => clearTimeout(id)
  }, [data, load])

  if (error) return <section className="page page-narrow"><p className="text-bad">{error}</p></section>
  if (!data) return <section className="page page-narrow"><p className="muted">{t('loading')}</p></section>
  const m = data.metrics || {}
  const tc = m.timecode || {}

  return (
    <section className="page">
      <div>
        <h1 className="page-title">{t('modelTitle')}</h1>
        <p className="lead left">{t('modelSub')}</p>
      </div>

      {data.system && (
        <section className="card">
          <header className="section-head"><h2>{t('systemTitle')}</h2><span className="muted small">{t('systemHint')}</span></header>
          <SystemTable system={data.system} />
        </section>
      )}

      {m.videos ? (
        <section className="card">
          <header className="section-head">
            <h2>{t('modelMetrics', { v: data.active_version })}</h2>
            <span className="muted small">{t('cvNote', { n: m.videos, folds: data.params?.folds })}</span>
          </header>
          <div className="tiles">
            <Tile label="Recall" value={pct(m.recall)} hint={`${m.tp}/${m.positives}`} />
            <Tile label="Precision" value={pct(m.precision)} />
            <Tile label="F1" value={num(m.f1)} />
            <Tile label="MCC" value={num(m.mcc)} />
            <Tile label="ROC-AUC" value={num(m.roc_auc)} />
            <Tile label="PR-AUC" value={num(m.pr_auc)} />
            <Tile label={t('falseAlarms')} value={`${m.fp}/${m.negatives}`} hint={`FPR ${pct(m.false_positive_rate)}`} />
            <Tile label="Brier" value={num(m.brier, 3)} hint={t('brierHint')} />
          </div>
          <div className="metrics-grid">
            <div>
              <h3 className="sub-h">{t('confusion')}</h3>
              <Confusion m={m} />
              <h3 className="sub-h">{t('localization')}</h3>
              <div className="tiles tiles-3">
                <Tile label={t('tcMedian')} value={sec(tc.median_sec)} />
                <Tile label="MAE" value={sec(tc.mae_sec)} />
                <Tile label={t('within3')} value={pct(tc.within_3s)} />
              </div>
              <h3 className="sub-h">{t('typeAcc')}</h3>
              <p className="small">{m.type?.n ? `${m.type.correct}/${m.type.n} (${pct(m.type.accuracy)})` : '—'} · <span className="muted">{t('typeHint')}</span></p>
            </div>
            {m.roc_curve && <RocChart points={m.roc_curve} auc={m.roc_auc} />}
          </div>
        </section>
      ) : <p className="muted">{t('noMetrics')}</p>}

      {m.fit && <FitCard fit={m.fit} search={m.search || []} params={data.params || {}} />}
      <LearningCard learning={data.learning} apiKey={apiKey} onChanged={load} />
      <Versions versions={data.versions} active={data.active_version} apiKey={apiKey} onChanged={load} />
    </section>
  )
}
