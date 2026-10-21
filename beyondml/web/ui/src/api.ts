// The gateway's REST surface, typed. Same-origin in production; Vite proxies /projects in dev.

export type Status = 'pending' | 'running' | 'done' | 'failed' | 'skipped' | 'n/a'

export interface BusEvent {
  ts: number
  agent: string
  kind: string
  data: Record<string, unknown>
}

export interface Metric {
  name: string
  value: number
}

export interface SuccessCheck {
  run: string | null
  expects: string[]
  metric: string | null
  min: number | null
}

export interface Step {
  id: string
  skill: string
  goal: string
  depends_on: string[]
  inputs: string[]
  outputs: string[]
  success_check: SuccessCheck
  status: Status
  model: string | null
  attempts: number
  summary: string
  last_failure: string
  started: number | null
  finished: number | null
  metric: Metric | null
}

export interface PlanView {
  goal: string
  metric: string
  budget: Record<string, unknown>
  steps: Step[]
  default_model: string
}

export interface PlanPayload {
  goal: string
  metric: string
  steps: { id: string; skill: string; goal: string; depends_on?: string[] }[]
}

export interface Prompt {
  event: 'question' | 'approval_request'
  agent: string
  id: string
  kind?: 'plan' | 'tool' | 'budget'
  question?: string
  tool?: string
  args?: unknown
  hint?: string | null
  plan?: PlanPayload
  reason?: string
}

export interface Summary {
  name: string
  root: string
  goal: string
  mode: string
  model: string
  running: boolean
  pending: Prompt[]
}

export interface ProjectRow {
  name: string
  root: string
  goal: string
  running: boolean
}

export interface AgentInfo {
  id: string
  status: Status
  skill: string | null
  goal: string
  inputs: string[]
  outputs: string[]
  tools: string[]
  model: string
  attempts: number
  instruction: string
  summary: string
  last_failure: string
}

export interface LedgerRecord {
  ts: number
  run_id: string
  agent: string
  script: string
  args: string[]
  exit_code: number | null
  timed_out: boolean
  seconds: number
  files_changed: string[]
  metrics: Record<string, unknown>
  data_hash?: string
}

export interface FileEntry {
  name: string
  dir: boolean
  size: number
}

export interface RunRequest {
  goal?: string
  data?: string[]
  mode?: 'ask' | 'auto' | 'autonomous'
  model?: string
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body)
    } catch {
      /* non-JSON error body: keep the status text */
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
})

const enc = (path: string) => path.split('/').map(encodeURIComponent).join('/')

export const api = {
  projects: () => request<ProjectRow[]>('/projects'),
  createProject: (path: string, goal: string) =>
    request<Summary>('/projects', json('POST', { path, goal: goal || null })),
  summary: (p: string) => request<Summary>(`/projects/${enc(p)}`),
  plan: (p: string) => request<PlanView>(`/projects/${enc(p)}/plan`),
  agent: (p: string, id: string) => request<AgentInfo>(`/projects/${enc(p)}/agents/${enc(id)}`),
  list: (p: string, path: string) =>
    request<{ path: string; entries: FileEntry[] }>(`/projects/${enc(p)}/files/${enc(path)}?list=1`),
  fileUrl: (p: string, path: string) => `/projects/${enc(p)}/files/${enc(path)}`,
  async file(p: string, path: string): Promise<{ text: string; type: string; size: number }> {
    const res = await fetch(api.fileUrl(p, path))
    if (!res.ok) throw new ApiError(res.status, `${res.status} ${res.statusText}`)
    const buf = await res.arrayBuffer()
    return {
      text: new TextDecoder('utf-8', { fatal: false }).decode(buf),
      type: res.headers.get('content-type') ?? '',
      size: buf.byteLength,
    }
  },
  async ledger(p: string): Promise<LedgerRecord[]> {
    try {
      const { text } = await api.file(p, 'experiments.jsonl')
      return text
        .split('\n')
        .filter((l) => l.trim())
        .map((l) => JSON.parse(l) as LedgerRecord)
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) return []
      throw e
    }
  },
  chat: (p: string, text: string) =>
    request<{ reply: string }>(`/projects/${enc(p)}/input`, json('POST', { type: 'chat', text })),
  answer: (p: string, id: string, answer: string | boolean) =>
    request<{ ok: boolean }>(`/projects/${enc(p)}/input`, json('POST', { type: 'answer', id, answer })),
  run: (p: string, body: RunRequest) => request<{ ok: boolean }>(`/projects/${enc(p)}/run`, json('POST', body)),
  stop: (p: string) => request<{ stopped: boolean }>(`/projects/${enc(p)}/stop`, { method: 'POST' }),
  async upload(p: string, file: File): Promise<{ path: string }> {
    const form = new FormData()
    form.append('file', file, file.name)
    return request<{ path: string }>(`/projects/${enc(p)}/upload`, { method: 'POST', body: form })
  },
  eventsUrl: (p: string, since?: number) =>
    `/projects/${enc(p)}/events${since ? `?since=${since}` : ''}`,
}
