const paths = {
  upload: 'M12 16V8m0 0-3.5 3.5M12 8l3.5 3.5M7 18a4 4 0 0 1-.6-7.96A6 6 0 0 1 18 9.5a3.5 3.5 0 0 1 .5 6.97',
  folder: 'M3 7.5A1.5 1.5 0 0 1 4.5 6H9l2 2h8.5A1.5 1.5 0 0 1 21 9.5v8a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5z',
  arrowRight: 'M5 12h14m-5-5 5 5-5 5',
  arrowLeft: 'M19 12H5m5-5-5 5 5 5',
  arrowUpRight: 'M8 16 16 8m-7 0h7v7',
  lock: 'M7 11V8a5 5 0 0 1 10 0v3M6 11h12v9H6z',
  shield: 'M12 3 5 6v5c0 4.5 3 8 7 10 4-2 7-5.5 7-10V6zm-3 9 2 2 4-4',
  frame: 'M4 8V5a1 1 0 0 1 1-1h3m8 0h3a1 1 0 0 1 1 1v3m0 8v3a1 1 0 0 1-1 1h-3m-8 0H5a1 1 0 0 1-1-1v-3m4-4h8',
  user: 'M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm-7 8a7 7 0 0 1 14 0',
  userCheck: 'M10 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm-7 8a7 7 0 0 1 11-5.7M16 18l2 2 4-4',
  sparkle: 'M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z',
  check: 'M5 12.5 10 17 19 7',
  checks: 'M3 12.5 7 16.5 15 8m-3 8.5L14 18l8-9',
  spinner: 'M12 3a9 9 0 1 0 9 9',
  circle: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z',
  alert: 'M12 9v4m0 4h.01M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z',
  checkCircle: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zm-4-9 3 3 5-6',
  info: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zm0-5v-5m0-3h.01',
  clock: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zm0-13v4l3 2',
  clipboard: 'M9 4h6v3H9zM8 5H6v16h12V5h-2m-7 8 2 2 4-4',
  expand: 'M4 9V4h5M20 9V4h-5M4 15v5h5m11-5v5h-5',
  plus: 'M12 5v14M5 12h14',
  play: 'M7 4v16l13-8z',
  x: 'M6 6l12 12M18 6 6 18',
  fileVideo: 'M14 3H6v18h12V7zm0 0v4h4M10 11v5l4-2.5z',
  fileAlert: 'M14 3H6v18h12V7zm0 0v4h4M12 11v3m0 3h.01',
  refresh: 'M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6',
  menu: 'M4 7h16M4 12h16M4 17h16',
  external: 'M14 4h6v6m0-6-9 9M18 14v6H4V6h6',
  download: 'M12 4v11m0 0-4-4m4 4 4-4M5 20h14',
  code: 'm8 8-4 4 4 4m8-8 4 4-4 4',
  sun: 'M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm0-14v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4',
  chart: 'M4 20V4m0 16h16M8 16l4-5 3 3 5-7',
  moon: 'M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z',
}

export default function Icon({ name, size = 18, className = '', strokeWidth = 1.8 }) {
  return (
    <svg className={`icon ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={paths[name]} />
    </svg>
  )
}
