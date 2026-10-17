const BASE = import.meta.env.VITE_API_BASE || ''

async function request(path, apiKey, init = {}) {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { ...(init.headers || {}), ...(apiKey ? { 'X-API-Key': apiKey } : {}) },
  })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join('; ') : body.detail
    throw new Error(detail || `HTTP ${res.status}`)
  }
  return body
}

const post = (path, apiKey, body) => request(path, apiKey, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: body ? JSON.stringify(body) : undefined,
})

export const health = () => request('/api/v1/health')
export const getConfig = () => request('/api/v1/config')
export const getJob = (jobId, apiKey) => request(`/api/v1/jobs/${jobId}`, apiKey)
export const getModel = () => request('/api/v1/model')
export const reviewJob = (jobId, apiKey, review) => post(`/api/v1/jobs/${jobId}/review`, apiKey, review)
export const retrain = (apiKey) => post('/api/v1/model/retrain', apiKey)
export const activateVersion = (id, apiKey) => post(`/api/v1/model/versions/${id}/activate`, apiKey)

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
      try { body = JSON.parse(xhr.responseText) } catch { body = {} }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body)
      else reject(new Error(body.detail || `HTTP ${xhr.status}`))
    }
    xhr.onerror = () => reject(new Error('Сеть недоступна: бэкенд не отвечает'))
    xhr.send(form)
  })
}
