// One live session: SSE events folded into screen state, plus the REST snapshots that depend on them.
// The fold mirrors beyondml/gateway/state.py; keep the two in step.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, type BusEvent, type PlanView, type Prompt, type Summary } from './api'

export interface StepTiming {
  status: string
  started: number | null
  finished: number | null
  reason: string
}

export interface ChatMessage {
  ts: number
  agent: 'user' | 'lead'
  text: string
}

export interface Generation {
  gen: number
  best_fitness: number
  mean_fitness: number
  diversity: number
  evals: number
  seconds: number
  best_params: Record<string, unknown>
}

export interface Trial {
  trial: number
  value: number
  best_value: number
  params: Record<string, unknown>
  seconds: number
}

export interface ChartData {
  title: string
  kind?: 'line' | 'bar' | 'scatter'
  x: (number | string)[]
  series: Record<string, number[]>
  path?: string
}

export interface Fold {
  steps: Record<string, StepTiming>
  pending: Record<string, Prompt>
  chat: ChatMessage[]
  generations: Record<string, Generation[]>
  trials: Record<string, Trial[]>
  charts: Record<string, ChartData[]>
  artifacts: Record<string, string[]>
  errors: { ts: number; agent: string; code?: string; message?: string }[]
  byAgent: Record<string, BusEvent[]>
  agents: string[]
  lastTs: number
}

const CHAT_AGENTS = new Set(['user', 'lead'])
const PER_AGENT_CAP = 3000

export function emptyFold(): Fold {
  return {
    steps: {}, pending: {}, chat: [], generations: {}, trials: {}, charts: {}, artifacts: {}, errors: [],
    byAgent: {}, agents: [], lastTs: 0,
  }
}

export function applyEvent(f: Fold, ev: BusEvent): void {
  const { agent, kind, ts } = ev
  const d = ev.data ?? {}
  f.lastTs = Math.max(f.lastTs, ts)
  if (!f.byAgent[agent]) {
    f.byAgent[agent] = []
    f.agents = [...f.agents, agent]
  }
  if (!(kind === 'message_delta' && d.partial)) {
    const list = f.byAgent[agent]
    list.push(ev)
    if (list.length > PER_AGENT_CAP) list.splice(0, list.length - PER_AGENT_CAP)
  }
  switch (kind) {
    case 'step_status': {
      if (typeof d.id !== 'string') break
      const step = (f.steps[d.id] ??= { status: 'pending', started: null, finished: null, reason: '' })
      step.status = (d.status as string) ?? 'pending'
      step.reason = (d.reason as string) ?? ''
      if (step.status === 'running') {
        step.started = ts
        step.finished = null
      } else if (step.status === 'done' || step.status === 'failed' || step.status === 'skipped') {
        step.finished = ts
      }
      break
    }
    case 'question':
    case 'approval_request': {
      if (typeof d.id !== 'string') break
      if ('answer' in d) delete f.pending[d.id]
      else f.pending[d.id] = { event: kind, agent, ...(d as object) } as Prompt
      break
    }
    case 'message_delta':
      if (CHAT_AGENTS.has(agent) && !d.partial) {
        f.chat.push({ ts, agent: agent as 'user' | 'lead', text: String(d.text ?? '') })
      }
      break
    case 'generation':
      (f.generations[agent] ??= []).push(d as unknown as Generation)
      break
    case 'trial':
      (f.trials[agent] ??= []).push(d as unknown as Trial)
      break
    case 'chart':
      (f.charts[agent] ??= []).push(d as unknown as ChartData)
      break
    case 'artifact': {
      if (typeof d.path !== 'string') break
      const paths = (f.artifacts[agent] ??= [])
      if (!paths.includes(d.path)) paths.push(d.path)
      break
    }
    case 'error':
      f.errors.push({ ts, agent, code: d.code as string, message: d.message as string })
      break
  }
}

const REFRESH_SUMMARY = new Set(['question', 'approval_request', 'step_status', 'error', 'plan_updated'])
const REFRESH_PLAN = new Set(['step_status', 'plan_updated', 'tool_result'])

export interface Session {
  name: string
  summary: Summary | null
  plan: PlanView | null
  planError: string | null
  fold: Fold
  version: number
  diskVersion: number // bumps when files or the ledger changed on disk
  connected: boolean
  pending: Prompt[]
  refreshSummary: () => void
  refreshPlan: () => void
}

const REFRESH_DISK = new Set(['artifact', 'step_status'])

export function useSession(name: string): Session {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [plan, setPlan] = useState<PlanView | null>(null)
  const [planError, setPlanError] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)
  const [version, setVersion] = useState(0)
  const [diskVersion, setDiskVersion] = useState(0)
  const foldRef = useRef<Fold>(emptyFold())
  const frame = useRef<number | null>(null)
  const timers = useRef<{ summary?: number; plan?: number; disk?: number }>({})

  const bump = useCallback(() => {
    if (frame.current != null) return
    frame.current = requestAnimationFrame(() => {
      frame.current = null
      setVersion((v) => v + 1)
    })
  }, [])

  const refreshSummary = useCallback(() => {
    api.summary(name).then(setSummary).catch(() => undefined)
  }, [name])

  const refreshPlan = useCallback(() => {
    api.plan(name)
      .then((p) => { setPlan(p); setPlanError(null) })
      .catch((e) => { setPlan(null); setPlanError(e?.status === 404 ? null : String(e.message ?? e)) })
  }, [name])

  const schedule = useCallback((which: 'summary' | 'plan' | 'disk', fn: () => void) => {
    const t = timers.current
    if (t[which]) window.clearTimeout(t[which])
    t[which] = window.setTimeout(fn, 300) // the executor saves plan.json before it emits; this just coalesces bursts
  }, [])

  useEffect(() => {
    foldRef.current = emptyFold()
    setVersion(0)
    refreshSummary()
    refreshPlan()
    const source = new EventSource(api.eventsUrl(name))
    source.onopen = () => setConnected(true)
    source.onerror = () => setConnected(false)
    source.onmessage = (msg) => {
      let ev: BusEvent
      try {
        ev = JSON.parse(msg.data) as BusEvent
      } catch {
        return
      }
      applyEvent(foldRef.current, ev)
      bump()
      if (REFRESH_SUMMARY.has(ev.kind)) schedule('summary', refreshSummary)
      const ranPython = ev.kind === 'tool_result' && ev.data?.name === 'run_python'
      if (REFRESH_PLAN.has(ev.kind) && (ev.kind !== 'tool_result' || ranPython)) schedule('plan', refreshPlan)
      if (REFRESH_DISK.has(ev.kind) || ranPython) schedule('disk', () => setDiskVersion((v) => v + 1))
    }
    const current = timers.current
    return () => {
      source.close()
      Object.values(current).forEach((t) => t && window.clearTimeout(t))
      if (frame.current != null) cancelAnimationFrame(frame.current)
      frame.current = null
    }
  }, [name, bump, refreshPlan, refreshSummary, schedule])

  const pending = useMemo(() => Object.values(foldRef.current.pending), [version]) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    name, summary, plan, planError, fold: foldRef.current, version, diskVersion, connected, pending, refreshSummary,
    refreshPlan,
  }
}

export function elapsed(started: number | null, finished: number | null, now = Date.now() / 1000): string {
  if (!started) return '—'
  const s = Math.max(0, (finished ?? now) - started)
  if (s < 60) return `${s.toFixed(1)}s`
  if (s < 3600) return `${Math.floor(s / 60)}m${String(Math.floor(s % 60)).padStart(2, '0')}s`
  return `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m`
}

export function fmt(v: unknown): string {
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return String(v)
    return Math.abs(v) < 1e4 ? v.toPrecision(4).replace(/\.?0+$/, '') : v.toExponential(3)
  }
  if (v == null) return '—'
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

export function useNow(active: boolean, everyMs = 1000): number {
  const [now, setNow] = useState(() => Date.now() / 1000)
  useEffect(() => {
    if (!active) return
    const id = window.setInterval(() => setNow(Date.now() / 1000), everyMs)
    return () => window.clearInterval(id)
  }, [active, everyMs])
  return now
}
