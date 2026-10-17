import { useCallback, useEffect, useRef, useState } from 'react'

export const ZONES = {
  cash_zone: { label: 'Кассовая зона', color: '#f59e0b', hint: 'рабочее место кассира' },
  client_zone: { label: 'Зона клиента', color: '#06b6d4', hint: 'место клиента у окна' },
}

export default function ZoneEditor({ videoUrl, zones, onChange }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [active, setActive] = useState('cash_zone')
  const [mode, setMode] = useState('loading')
  const ready = mode !== 'loading'

  useEffect(() => {
    setMode('loading')
    const id = setTimeout(() => setMode((m) => (m === 'loading' ? 'grid' : m)), 3000)
    return () => clearTimeout(id)
  }, [videoUrl])

  const draw = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!canvas || !ready) return
    const hasFrame = mode === 'frame' && video?.videoWidth
    const w = canvas.clientWidth
    const h = Math.round(hasFrame ? (w * video.videoHeight) / video.videoWidth : (w * 9) / 16)
    canvas.width = w * devicePixelRatio
    canvas.height = h * devicePixelRatio
    canvas.style.height = `${h}px`
    const ctx = canvas.getContext('2d')
    ctx.scale(devicePixelRatio, devicePixelRatio)
    if (hasFrame) {
      ctx.drawImage(video, 0, 0, w, h)
    } else {
      ctx.fillStyle = '#111827'
      ctx.fillRect(0, 0, w, h)
      ctx.strokeStyle = '#374151'
      ctx.lineWidth = 1
      for (let i = 1; i < 10; i++) {
        ctx.beginPath(); ctx.moveTo((w * i) / 10, 0); ctx.lineTo((w * i) / 10, h); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(0, (h * i) / 10); ctx.lineTo(w, (h * i) / 10); ctx.stroke()
      }
    }

    for (const [name, pts] of Object.entries(zones)) {
      if (!pts?.length) continue
      const { color } = ZONES[name] || { color: '#999' }
      ctx.beginPath()
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(x * w, y * h) : ctx.moveTo(x * w, y * h)))
      if (pts.length >= 3) ctx.closePath()
      ctx.fillStyle = color + '33'
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      if (pts.length >= 3) ctx.fill()
      ctx.stroke()
      pts.forEach(([x, y]) => {
        ctx.beginPath()
        ctx.arc(x * w, y * h, 4, 0, Math.PI * 2)
        ctx.fillStyle = color
        ctx.fill()
      })
    }
  }, [zones, mode, ready])

  useEffect(() => { draw() }, [draw])
  useEffect(() => {
    const onResize = () => ready && draw()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [ready, draw])

  const addPoint = (e) => {
    const rect = canvasRef.current.getBoundingClientRect()
    const x = +((e.clientX - rect.left) / rect.width).toFixed(4)
    const y = +((e.clientY - rect.top) / rect.height).toFixed(4)
    onChange({ ...zones, [active]: [...(zones[active] || []), [x, y]] })
  }

  const update = (name, pts) => {
    const next = { ...zones }
    if (pts.length) next[name] = pts
    else delete next[name]
    onChange(next)
  }

  return (
    <div className="zone-editor">
      <video
        ref={videoRef}
        src={videoUrl}
        muted
        playsInline
        preload="auto"
        style={{ display: 'none' }}
        onLoadedData={(e) => { e.currentTarget.currentTime = Math.min(1, e.currentTarget.duration / 2) }}
        onSeeked={(e) => e.currentTarget.videoWidth && setMode('frame')}
        onError={() => setMode('grid')}
      />
      <div className="zone-toolbar">
        {Object.entries(ZONES).map(([name, z]) => (
          <button
            key={name}
            type="button"
            className={`chip${active === name ? ' is-active' : ''}`}
            style={{ '--chip': z.color }}
            onClick={() => setActive(name)}
          >
            <span className="chip-dot" />
            {z.label}
            <span className="chip-count">{zones[name]?.length || 0}</span>
          </button>
        ))}
        <span className="toolbar-spacer" />
        <button type="button" className="btn-ghost" onClick={() => update(active, (zones[active] || []).slice(0, -1))}>
          Отменить точку
        </button>
        <button type="button" className="btn-ghost" onClick={() => update(active, [])}>
          Очистить зону
        </button>
      </div>
      <canvas ref={canvasRef} className="zone-canvas" onClick={addPoint} />
      {mode === 'loading' && <p className="muted">Загружаем кадр из видео…</p>}
      {mode === 'grid' && (
        <p className="muted small">
          Браузер не смог показать кадр (часто так с HEVC .mov в Chrome). Размечайте по сетке 10×10:
          координаты задаются в долях кадра, на результат это не влияет.
        </p>
      )}
      <p className="muted small">
        Кликайте по кадру, чтобы обвести {ZONES[active].hint}. Нужно минимум 3 точки.
        Без зоны клиента правило «кассир покинул рабочее место» не проверяется.
      </p>
    </div>
  )
}
