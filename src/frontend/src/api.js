const API_BASE = import.meta.env.VITE_API_BASE || ''

function apiUrl(path) {
  return `${API_BASE}${path}`
}

export async function fetchModels() {
  const res = await fetch(apiUrl('/api/models'))
  if (!res.ok) throw new Error(`模型列表获取失败：${res.status}`)
  return res.json()
}

export async function predictAudio(file, modelName) {
  const form = new FormData()
  form.append('model_name', modelName)
  form.append('file', file)
  const res = await fetch(apiUrl('/api/predict'), {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`识别请求失败：${res.status} ${text}`)
  }
  return res.json()
}

export async function compareAudio(file, modelNames) {
  const form = new FormData()
  form.append('model_names', modelNames.join(','))
  form.append('file', file)
  const res = await fetch(apiUrl('/api/compare'), {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`对比请求失败：${res.status} ${text}`)
  }
  return res.json()
}
