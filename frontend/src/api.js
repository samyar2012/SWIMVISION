const API_BASE = import.meta.env.VITE_API_URL || ''

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail || body.message || 'SwimVision could not complete that request.')
  return body
}

export async function createAnalysis({ file, stroke, distance, pool, officialTime }) {
  const form = new FormData()
  form.append('video', file)
  form.append('stroke', stroke)
  form.append('distance', String(distance))
  form.append('pool', pool)
  if (officialTime.trim()) form.append('official_time', officialTime.trim())
  return request('/api/analyses', { method: 'POST', body: form })
}

export const getFrameUrl = (analysisId) => `${API_BASE}/api/analyses/${analysisId}/frame`
export const selectSwimmer = (analysisId, point) => request(`/api/analyses/${analysisId}/selection`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(point),
})
export const startAnalysis = (analysisId, officialTime) => request(`/api/analyses/${analysisId}/run`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ official_time_s: officialTime || null }),
})
export const getAnalysisStatus = (analysisId) => request(`/api/analyses/${analysisId}/status`)
export const getAnalysis = (analysisId) => request(`/api/analyses/${analysisId}`)
export const askCoach = (analysisId, question) => request(`/api/analyses/${analysisId}/coach`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }),
})
