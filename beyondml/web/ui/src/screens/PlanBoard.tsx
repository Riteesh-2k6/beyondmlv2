// Plan board: the steps as dense rows on the left, the dependency graph on the right; the run form when there is none.
import { useState, type FormEvent } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { api, type RunRequest } from '../api'
import { DepGraph } from '../components/DepGraph'
import { StepCard } from '../components/StepCard'
import { Button, Chip, Empty, PanelTitle, field } from '../components/ui'
import { useProject } from '../context'
import { useNow } from '../session'

function RunForm({ onStarted, resumable }: { onStarted: () => void; resumable: boolean }) {
  const session = useProject()
  const [goal, setGoal] = useState('')
  const [data, setData] = useState('')
  const [uploaded, setUploaded] = useState<string[]>([])
  const [mode, setMode] = useState<RunRequest['mode'] | ''>('')
  const [model, setModel] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function upload(file: File | undefined) {
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      const { path } = await api.upload(session.name, file)
      setUploaded((u) => (u.includes(path) ? u : [...u, path]))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function start(e: FormEvent, resume = false) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    const body: RunRequest = resume ? {} : { goal: goal.trim() || undefined, data: data.trim() ? [data.trim()] : [] }
    if (mode) body.mode = mode
    if (model.trim()) body.model = model.trim()
    try {
      await api.run(session.name, body)
      onStarted()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={start} className="max-w-2xl space-y-3 p-4">
      <div>
        <label htmlFor="goal" className="mb-1 block text-xs text-muted">goal</label>
        <textarea
          id="goal"
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          rows={2}
          placeholder="predict churn from the usage and billing columns; optimise F1"
          className={`${field} h-auto resize-y py-1.5`}
        />
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <label htmlFor="data" className="mb-1 block text-xs text-muted">data on this machine</label>
          <input id="data" value={data} onChange={(e) => setData(e.target.value)} placeholder="C:\data\churn.csv or a folder" className={field} />
        </div>
        <div>
          <label htmlFor="upload" className="mb-1 block text-xs text-muted">or upload into data/raw</label>
          <input id="upload" type="file" onChange={(e) => void upload(e.target.files?.[0])} disabled={busy}
            className="block w-full text-xs text-muted file:mr-2 file:h-7 file:rounded-[2px] file:border file:border-border-sub file:bg-surface file:px-2 file:text-xs file:text-fg hover:file:border-muted" />
          {uploaded.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">{uploaded.map((p) => <Chip key={p} tone="accent">{p}</Chip>)}</div>
          )}
        </div>
        <div>
          <label htmlFor="mode" className="mb-1 block text-xs text-muted">mode</label>
          <select id="mode" value={mode} onChange={(e) => setMode(e.target.value as RunRequest['mode'] | '')} className={field}>
            <option value="">project default ({session.summary?.mode ?? 'auto'})</option>
            <option value="ask">ask: confirm every run and install</option>
            <option value="auto">auto: confirm installs and network</option>
            <option value="autonomous">autonomous: never ask</option>
          </select>
        </div>
        <div>
          <label htmlFor="model" className="mb-1 block text-xs text-muted">model for every agent</label>
          <input id="model" value={model} onChange={(e) => setModel(e.target.value)} placeholder={session.summary?.model ?? 'ollama_chat/qwen2.5:14b'} className={`${field} font-mono`} />
        </div>
      </div>
      <div className="flex items-center gap-2 pt-1">
        <Button type="submit" kind="primary" disabled={busy || (!goal.trim() && !resumable)}>
          {resumable && !goal.trim() ? 'Resume the plan' : 'Plan and run'}
        </Button>
        {resumable && goal.trim() && <Button onClick={() => void start(new Event('submit') as unknown as FormEvent, true)} disabled={busy}>Resume instead</Button>}
        {error && <span className="text-xs text-danger">{error}</span>}
      </div>
      <p className="text-xs text-muted">
        Intake profiles the data and writes DATA.md; the planner writes plan.json from the available skills. Unless the
        mode is autonomous you approve the plan before any step runs.
      </p>
    </form>
  )
}

export function PlanBoard() {
  const session = useProject()
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const { plan, summary, fold } = session
  const anyRunning = !!plan?.steps.some((s) => s.status === 'running')
  const now = useNow(anyRunning)
  const showForm = params.get('new') === '1'
  const open = (id: string) => navigate(`/p/${encodeURIComponent(session.name)}/agents/${encodeURIComponent(id)}`)

  if (!plan && summary?.running) {
    return (
      <div className="p-4">
        <Empty title="Planning…">Intake and the planner are working; the plan appears here the moment plan.json is written.</Empty>
        <div className="mt-3 space-y-2">{[0, 1, 2].map((i) => <div key={i} className="skeleton h-12" />)}</div>
      </div>
    )
  }
  if (!plan || showForm) {
    const unfinished = !!plan && plan.steps.some((s) => s.status === 'pending' || s.status === 'running')
    return (
      <div className="flex h-full flex-col overflow-y-auto">
        <PanelTitle right={showForm && <button type="button" className="hover:text-fg" onClick={() => setParams({})}>back to the plan</button>}>
          {plan ? 'New run' : 'No plan yet'}
        </PanelTitle>
        {session.planError && <p className="px-4 pt-3 text-xs text-danger">plan.json could not be read: {session.planError}</p>}
        <RunForm resumable={unfinished && !summary?.running} onStarted={() => { setParams({}); session.refreshSummary() }} />
      </div>
    )
  }

  const counts = plan.steps.reduce<Record<string, number>>((acc, s) => ((acc[s.status] = (acc[s.status] ?? 0) + 1), acc), {})
  const summaryLine = ['done', 'running', 'pending', 'failed', 'skipped'].filter((k) => counts[k]).map((k) => `${counts[k]} ${k}`).join(' · ')
  const finished = plan.steps.every((s) => s.status === 'done' || s.status === 'failed' || s.status === 'skipped')

  return (
    <div className="flex h-full min-h-0 flex-col lg:flex-row">
      <section className="flex min-h-0 flex-1 flex-col lg:border-r lg:border-border-sub" aria-label="plan steps">
        <PanelTitle right={<>
          <span className="font-mono">{summaryLine}</span>
          <span>metric {plan.metric}</span>
          {!summary?.running && (
            <button type="button" onClick={() => setParams({ new: '1' })} className="text-orange hover:underline">
              {finished ? 'new run' : 'resume'}
            </button>
          )}
        </>}>
          {plan.goal}
        </PanelTitle>
        <div className="min-h-0 flex-1 overflow-y-auto">
          {plan.steps.map((s) => (
            <StepCard key={s.id} step={s} defaultModel={plan.default_model} now={now} onOpen={open} />
          ))}
          {fold.errors.length > 0 && (
            <div className="border-b border-border-sub px-3 py-2 text-xs text-danger">
              {fold.errors.slice(-3).map((e, i) => <p key={i}>{e.agent}: {e.code} {e.message}</p>)}
            </div>
          )}
        </div>
      </section>
      <aside className="flex max-h-72 shrink-0 flex-col border-t border-border-sub lg:max-h-none lg:w-[38%] lg:border-t-0" aria-label="dependency graph">
        <PanelTitle right={<span>{plan.steps.length} steps</span>}>Dependencies</PanelTitle>
        <div className="min-h-0 flex-1 overflow-auto p-3">
          <DepGraph steps={plan.steps} onOpen={open} />
        </div>
      </aside>
    </div>
  )
}
