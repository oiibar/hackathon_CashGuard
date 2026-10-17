import Icon from './Icons.jsx'
import { useT } from '../i18n.js'

export function HowView() {
  const t = useT()
  return (
    <section className="page page-narrow">
      <h1 className="page-title">{t('howTitle')}</h1>
      <ol className="how">
        {t('how').map(([title, text], i) => (
          <li key={title} className="card">
            <span className="how-num">{i + 1}</span>
            <div><strong>{title}</strong><p className="muted">{text}</p></div>
          </li>
        ))}
      </ol>
      <div className="note-card">
        <Icon name="shield" size={20} className="accent" />
        <div><strong>{t('privateT')}</strong><span>{t('howLocal')}</span></div>
      </div>
    </section>
  )
}

export function TypesView({ rules }) {
  const t = useT()
  return (
    <section className="page page-narrow">
      <h1 className="page-title">{t('typesTitle')}</h1>
      <p className="lead left">{t('typesSub')}</p>
      <div className="types">
        {(rules || []).map((r) => (
          <article key={r.type} className={`card type${r.active === false ? ' is-off' : ''}`}>
            <header>
              <h3>{r.title}</h3>
              {r.active === false && <span className="pill pill-soft">{t('needsModel')}</span>}
            </header>
            <p className="muted small">{r.why}</p>
            <div className="legal-box">
              <span className="legal-label">{t('legalLabel')}</span>
              {r.legal_basis.map((b, i) => (
                <div key={i} className="legal-item">
                  <strong>{b.short}, {b.clause}</strong>
                  <q>{b.quote}</q>
                  <a href={b.url} target="_blank" rel="noreferrer">{t('openAct')}<Icon name="external" size={13} /></a>
                </div>
              ))}
            </div>
            <code className="xsmall muted">{r.type}</code>
          </article>
        ))}
      </div>
    </section>
  )
}
