// Thin client for the FastAPI backend. Same origin by default (Vite proxy in
// dev, FastAPI static mount in prod); set VITE_API_BASE to point elsewhere.
const BASE = import.meta.env.VITE_API_BASE || ''

async function request(path, apiKey, init = {}) {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { ...(init.headers || {}), ...(apiKey ? { 'X-API-Key': apiKey } : {}) },
  })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`)
  return body
}

export const health = () => request('/api/v1/health')
export const getConfig = (apiKey) => request('/api/v1/config', apiKey)
export const getJob = (jobId, apiKey) => request(`/api/v1/jobs/${jobId}`, apiKey)

// XHR instead of fetch: fetch has no upload progress events, and .mov files
// from a phone can be hundreds of MB.
export function analyze(file, { apiKey, zones, profile }, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('file', file)
    if (zones) form.append('zones', JSON.stringify(zones))
    else if (profile) form.append('profile', profile)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', BASE + '/api/v1/analyze')
    xhr.setRequestHeader('X-API-Key', apiKey)
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total)
    xhr.onload = () => {
      let body = {}
      try { body = JSON.parse(xhr.responseText) } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body)
      else reject(new Error(body.detail || `HTTP ${xhr.status}`))
    }
    xhr.onerror = () => reject(new Error('Сеть недоступна: бэкенд не отвечает'))
    xhr.send(form)
  })
}
