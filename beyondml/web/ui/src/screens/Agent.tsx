// Agent screen: the exact instruction, the live stream, the outputs, and the step's own view.
import clsx from 'clsx'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api, type AgentInfo } from '../api'
import { ChartEvent } from '../components/ChartEvent'
import { EvolveView } from '../components/EvolveView'
import { LiveLog } from '../components/LiveLog'
import { Chip, Empty, StatusMark, StatusWord } from '../components/ui'
import { useProject } from '../context'
import { fmt } from '../session'

const FIXED = ['intake', 'planner', 'executor', 'lead', 'gateway']
type Tab = 'instruction' | 'live' | 'outputs' | 'evolve'

export function Agent() {
  const session = useProject()
  const { id } = useParams()
  const navigate = useNavigate()
  const { plan, fold, version } = session
  const [info, setInfo] = useState<AgentInfo | null>(null)
  const [missing, setMissing] = useState(false)
  const [tab, setTab] = useState<Tab>('instruction')

  const agents = useMemo(() => {
    const steps = plan?.steps.map((s) => ({ id: s.id, status: s.status as string })) ?? []
    const others = FIXED.filter((a) => fold.agents.includes(a)).map((a) => ({ id: a, status: fold.steps[a]?.status ?? 'n/a' }))
    return [...steps, ...others]
  }, [plan, fold, version]) // eslint-disable-line react-hooks/exhaustive-deps

  const statusKey = id ? `${fold.steps[id]?.status ?? ''}:${plan?.steps.find((s) => s.id === id)?.status ?? ''}` : ''
  useEffect(() => {
    if (!id) return
    let live = true
    api.agent(session.name, id)
      .then((a) => { if (live) { setInfo(a); setMissing(false) } })
      .catch(() => { if (live) { setInfo(null); setMissing(true) } })
    return () => { live = false }
  }, [session.name, id, statusKey])

  useEffect(() => {
    if (!id && agents.length) navigate(`/p/${encodeURIComponent(session.name)}/agents/${encodeURIComponent(agents[0].id)}`, { replace: true })
  }, [id, agents, navigate, session.name])

  const events = id ? fold.byAgent[id] ?? [] : []
  const gens = id ? fold.generations[id] ?? [] : []
  const trials = id ? fold.trials[id] ?? [] : []
  const hasEvolve = gens.length > 0 || trials.length > 0
  const charts = id ? fold.charts[id] ?? [] : []
  const files = useMemo(() => {
    const declared = info?.outputs ?? []
    const written = id ? fold.artifacts[id] ?? [] : []
    return [...new Set([...declared, ...written])]
  }, [info, id, fold.artifacts, version]) // eslint-disable-line react-hooks/exhaustive-deps
  const metrics = useMemo(() => latestMetrics(events), [events, version]) // eslint-disable-line react-hooks/exhaustive-deps
  const activeTab: Tab = tab === 'evolve' && !hasEvolve ? 'instruction' : tab

  if (!agents.length) {
    return <Empty title="No agents yet">Agents appear here as the plan is written: one per step, plus intake, the planner and the lead.</Empty>
  }

  const tabs: { key: Tab; label: string }[] = [
    { key: 'instruction', label: 'Instruction' },
    { key: 'live', label: `Live${events.length ? ` · ${events.length}` : ''}` },
    { key: 'outputs', label: `Outputs${files.length + charts.length ? ` · ${files.length + charts.length}` : ''}` },
    ...(hasEvolve ? [{ key: 'evolve' as Tab, label: trials.length && !gens.length ? 'Tune' : 'Evolve' }] : []),
  ]

  return (
    <div className="flex h-full min-h-0 flex-col md:flex-row">
      <nav aria-label="agents" className="shrink-0 border-b border-border-sub md:w-52 md:border-b-0 md:border-r">
        <select
          className="m-2 h-7 w-[calc(100%-1rem)] rounded-[2px] border border-border-sub bg-obsidian px-1 font-mono text-xs md:hidden"
          value={id ?? ''}
          onChange={(e) => navigate(`/p/${encodeURIComponent(session.name)}/agents/${encodeURIComponent(e.target.value)}`)}
          aria-label="choose an agent"
        >
          {agents.map((a) => <option key={a.id} value={a.id}>{a.id} · {a.status}</option>)}
        </select>
        <ul className="hidden md:block">
          {agents.map((a) => (
            <li key={a.id}>
              <button
                type="button"
                onClick={() => navigate(`/p/${encodeURIComponent(session.name)}/agents/${encodeURIComponent(a.id)}`)}
                className={clsx('flex w-full items-center gap-2 px-3 py-1.5 text-left font-mono text-xs transition-colors duration-150 hover:bg-surface',
                  a.id === id ? 'bg-surface text-fg' : 'text-muted')}
                aria-current={a.id === id ? 'page' : undefined}
              >
                <StatusMark status={a.status} size={10} />
                <span className="truncate">{a.id}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>
      <section className="flex min-h-0 flex-1 flex-col" aria-label={`agent ${id ?? ''}`}>
        <header className="flex min-h-10 flex-wrap items-center gap-x-3 gap-y-1 border-b border-border-sub bg-surface px-3 py-1.5">
          <h2 className="font-mono text-sm font-medium text-fg">{id}</h2>
          {info && <StatusWord status={info.status} />}
          {info?.skill && <span className="text-xs text-muted">{info.skill}</span>}
          {info && <span className="font-mono text-xs text-muted">{info.model}</span>}
          {info && info.attempts > 0 && <span className="text-xs text-muted">attempt {info.attempts}</span>}
          {missing && <span className="text-xs text-danger">the gateway does not know this agent</span>}
          <div role="tablist" className="ml-auto flex gap-0.5">
            {tabs.map((t) => (
              <button
                key={t.key}
                role="tab"
                aria-selected={activeTab === t.key}
                onClick={() => setTab(t.key)}
                className={clsx('h-7 rounded-[2px] px-2.5 text-xs transition-colors duration-150',
                  activeTab === t.key ? 'bg-obsidian text-orange' : 'text-muted hover:text-fg')}
              >
                {t.label}
              </button>
            ))}
          </div>
        </header>
        {info?.goal && activeTab === 'instruction' && <p className="border-b border-border-sub px-3 py-1.5 text-xs text-fg/80">{info.goal}</p>}

        {activeTab === 'instruction' && (
          <div className="min-h-0 flex-1 overflow-y-auto">
            <dl className="grid grid-cols-[6rem_1fr] gap-y-1 border-b border-border-sub px-3 py-2 text-xs">
              <dt className="text-muted">skill</dt><dd className="font-mono">{info?.skill ?? '—'}</dd>
              <dt className="text-muted">inputs</dt><dd className="flex flex-wrap gap-1">{info?.inputs.length ? info.inputs.map((i) => <Chip key={i}>{i}</Chip>) : '—'}</dd>
              <dt className="text-muted">outputs</dt><dd className="flex flex-wrap gap-1">{info?.outputs.length ? info.outputs.map((o) => <Chip key={o}>{o}</Chip>) : '—'}</dd>
              <dt className="text-muted">tools</dt><dd className="flex flex-wrap gap-1">{info?.tools.length ? info.tools.map((t) => <Chip key={t} tone="accent">{t}</Chip>) : '—'}</dd>
            </dl>
            {info?.instruction ? (
              <pre className="whitespace-pre-wrap break-words px-3 py-3 font-mono text-xs leading-5 text-fg/90">{info.instruction}</pre>
            ) : (
              <p className="px-3 py-3 text-xs text-muted">
                {info?.skill ? 'The instruction is written to agents/<id>/instruction.md the moment the step starts.' : 'This agent has a fixed role and no written instruction file.'}
              </p>
            )}
          </div>
        )}

        {activeTab === 'live' && <LiveLog events={events} />}

        {activeTab === 'outputs' && (
          <div className="min-h-0 flex-1 overflow-y-auto p-3 text-sm">
            <h3 className="mb-1 text-xs font-medium text-fg">files</h3>
            {files.length ? (
              <ul className="mb-4 space-y-0.5 font-mono text-xs">
                {files.map((f) => (
                  <li key={f}>
                    <button type="button" className="text-fg/90 hover:text-orange hover:underline" onClick={() => navigate(`/p/${encodeURIComponent(session.name)}/files/${f}`)}>{f}</button>
                  </li>
                ))}
              </ul>
            ) : <p className="mb-4 text-xs text-muted">none yet</p>}
            <h3 className="mb-1 text-xs font-medium text-fg">metrics</h3>
            {Object.keys(metrics).length ? (
              <table className="mb-4 font-mono text-xs"><tbody>
                {Object.entries(metrics).map(([k, v]) => (
                  <tr key={k} className="border-b border-border-sub/60"><td className="py-1 pr-6 text-muted">{k}</td><td className="py-1 tabular-nums text-fg">{fmt(v)}</td></tr>
                ))}
              </tbody></table>
            ) : <p className="mb-4 text-xs text-muted">none reported yet; run_python collects metrics.json</p>}
            <h3 className="mb-1 text-xs font-medium text-fg">charts · {charts.length}</h3>
            {charts.length ? charts.map((c, i) => <div key={i} className="mb-3"><ChartEvent chart={c} /></div>) : <p className="text-xs text-muted">none yet</p>}
          </div>
        )}

        {activeTab === 'evolve' && (
          <div className="min-h-0 flex-1 overflow-y-auto"><EvolveView generations={gens} trials={trials} /></div>
        )}
      </section>
    </div>
  )
}

/** The latest metrics this agent's run_python results carried. */
function latestMetrics(events: { kind: string; data: Record<string, unknown> }[]): Record<string, unknown> {
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i]
    if (ev.kind !== 'tool_result' || ev.data.name !== 'run_python') continue
    const resp = ev.data.response as Record<string, unknown> | undefined
    const m = resp?.metrics
    if (m && typeof m === 'object' && Object.keys(m as object).length) return m as Record<string, unknown>
  }
  return {}
}
