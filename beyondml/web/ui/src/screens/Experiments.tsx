// Experiments: the ledger (experiments.jsonl) as a table, newest first.
import clsx from 'clsx'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type LedgerRecord } from '../api'
import { Chip, Empty, PanelTitle, timeOf } from '../components/ui'
import { useProject } from '../context'
import { fmt } from '../session'

export function Experiments() {
  const session = useProject()
  const navigate = useNavigate()
  const [records, setRecords] = useState<LedgerRecord[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [agent, setAgent] = useState('')
  const [open, setOpen] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    api.ledger(session.name)
      .then((r) => { if (live) { setRecords(r); setError(null) } })
      .catch((e) => { if (live) setError(e instanceof Error ? e.message : String(e)) })
    return () => { live = false }
  }, [session.name, session.diskVersion])

  const agents = useMemo(() => [...new Set((records ?? []).map((r) => r.agent))].sort(), [records])
  const rows = useMemo(() => (records ?? []).filter((r) => !agent || r.agent === agent).slice().reverse(), [records, agent])

  if (records === null && !error) return <div className="space-y-2 p-3">{[0, 1, 2, 3].map((i) => <div key={i} className="skeleton h-7" />)}</div>

  return (
    <div className="flex h-full min-h-0 flex-col">
      <PanelTitle right={<>
        <span>{records?.length ?? 0} runs from experiments.jsonl</span>
        {agents.length > 1 && (
          <select value={agent} onChange={(e) => setAgent(e.target.value)} className="h-6 rounded-[2px] border border-border-sub bg-obsidian px-1 font-mono text-2xs text-fg" aria-label="filter by agent">
            <option value="">all agents</option>
            {agents.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        )}
      </>}>
        Experiments
      </PanelTitle>
      {error && <p className="px-3 py-2 text-xs text-danger">{error}</p>}
      {records && records.length === 0 && (
        <Empty title="No experiments yet">
          Every run_python call lands here with its exit code, duration, the files it changed and the metrics it wrote to
          BEYONDML_METRICS. The executor's success checks land here too.
        </Empty>
      )}
      {rows.length > 0 && (
        <div className="min-h-0 flex-1 overflow-auto">
          <table className="w-full min-w-[720px] border-collapse font-mono text-xs">
            <thead className="sticky top-0 bg-obsidian">
              <tr className="border-b border-border-sub text-left text-muted">
                <th className="px-3 py-1.5 font-normal">time</th>
                <th className="px-3 py-1.5 font-normal">agent</th>
                <th className="px-3 py-1.5 font-normal">script</th>
                <th className="px-3 py-1.5 text-right font-normal">exit</th>
                <th className="px-3 py-1.5 text-right font-normal">seconds</th>
                <th className="px-3 py-1.5 font-normal">metrics</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const failed = r.timed_out || (r.exit_code ?? 0) !== 0
                const key = `${r.run_id}-${r.ts}`
                const expanded = open === key
                return (
                  <FragmentRow key={key} expanded={expanded} onToggle={() => setOpen(expanded ? null : key)} record={r} failed={failed}
                    onAgent={() => navigate(`/p/${encodeURIComponent(session.name)}/agents/${encodeURIComponent(r.agent)}`)} />
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function FragmentRow({ record: r, failed, expanded, onToggle, onAgent }: {
  record: LedgerRecord; failed: boolean; expanded: boolean; onToggle: () => void; onAgent: () => void
}) {
  const metrics = Object.entries(r.metrics ?? {}).filter(([, v]) => typeof v !== 'object')
  return (
    <>
      <tr onClick={onToggle} className={clsx('cursor-pointer border-b border-border-sub/60 transition-colors duration-150 hover:bg-surface', expanded && 'bg-surface')} aria-expanded={expanded}>
        <td className="whitespace-nowrap px-3 py-1.5 tabular-nums text-muted">{timeOf(r.ts)}</td>
        <td className="px-3 py-1.5">
          <button type="button" onClick={(e) => { e.stopPropagation(); onAgent() }} className="text-fg hover:text-orange hover:underline">{r.agent}</button>
        </td>
        <td className="max-w-[18rem] truncate px-3 py-1.5 text-fg/90">{[r.script, ...(r.args ?? [])].join(' ')}</td>
        <td className={clsx('px-3 py-1.5 text-right tabular-nums', failed ? 'text-danger' : 'text-success')}>{r.timed_out ? 'timeout' : r.exit_code ?? '—'}</td>
        <td className="px-3 py-1.5 text-right tabular-nums text-fg/90">{(r.seconds ?? 0).toFixed(1)}</td>
        <td className="px-3 py-1.5">
          <div className="flex flex-wrap gap-1">
            {metrics.length ? metrics.map(([k, v]) => <Chip key={k}>{k}=<span className="text-fg">{fmt(v)}</span></Chip>) : <span className="text-muted">—</span>}
          </div>
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-border-sub bg-surface/60">
          <td colSpan={6} className="px-3 py-2">
            <dl className="grid grid-cols-[7rem_1fr] gap-y-1 text-muted">
              <dt>run id</dt><dd className="text-fg/90">{r.run_id}</dd>
              <dt>files changed</dt><dd className="text-fg/90">{r.files_changed?.length ? r.files_changed.join(', ') : 'none'}</dd>
              <dt>metrics</dt><dd className="whitespace-pre-wrap text-fg/90">{Object.keys(r.metrics ?? {}).length ? JSON.stringify(r.metrics, null, 1) : 'none'}</dd>
              {r.data_hash && <><dt>data hash</dt><dd className="truncate text-fg/90">{r.data_hash}</dd></>}
            </dl>
          </td>
        </tr>
      )}
    </>
  )
}
