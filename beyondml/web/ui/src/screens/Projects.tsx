// Projects: what this gateway has open, and a way to open or create another.
import { useEffect, useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, type ProjectRow } from '../api'
import { Button, StatusMark, field } from '../components/ui'

export function Projects() {
  const navigate = useNavigate()
  const [rows, setRows] = useState<ProjectRow[] | null>(null)
  const [path, setPath] = useState('')
  const [goal, setGoal] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.projects().then(setRows).catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  useEffect(() => {
    if (rows && rows.length === 1 && !error) navigate(`/p/${encodeURIComponent(rows[0].name)}`, { replace: true })
  }, [rows, error, navigate])

  async function create(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const s = await api.createProject(path.trim(), goal.trim())
      navigate(`/p/${encodeURIComponent(s.name)}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="text-base font-medium text-fg"><span className="text-orange">BeyondML</span> projects</h1>
      <p className="mt-1 text-sm text-muted">A project is a folder: the data, the plan, each agent's instruction and events, and the code that ships.</p>
      {error && <p className="mt-3 text-xs text-danger">{error}</p>}
      <ul className="mt-6 border-t border-border-sub">
        {rows?.map((r) => (
          <li key={r.name} className="border-b border-border-sub">
            <Link to={`/p/${encodeURIComponent(r.name)}`} className="flex items-center gap-3 px-2 py-2.5 transition-colors duration-150 hover:bg-surface">
              <StatusMark status={r.running ? 'running' : 'n/a'} />
              <span className="font-mono text-sm text-fg">{r.name}</span>
              <span className="min-w-0 flex-1 truncate text-sm text-muted">{r.goal || 'no goal yet'}</span>
              <span className="hidden truncate font-mono text-2xs text-muted sm:block" style={{ maxWidth: '18rem' }}>{r.root}</span>
            </Link>
          </li>
        ))}
        {rows && rows.length === 0 && <li className="px-2 py-3 text-sm text-muted">The gateway has no project open. Create or open one below.</li>}
        {rows === null && !error && <li className="skeleton my-2 h-9" />}
      </ul>
      <form onSubmit={create} className="mt-8 space-y-3">
        <h2 className="text-sm font-medium text-fg">Open or create a project</h2>
        <div>
          <label htmlFor="path" className="mb-1 block text-xs text-muted">absolute folder path on this machine</label>
          <input id="path" value={path} onChange={(e) => setPath(e.target.value)} placeholder="C:\Users\you\churn-model" className={`${field} font-mono`} required />
        </div>
        <div>
          <label htmlFor="pgoal" className="mb-1 block text-xs text-muted">goal, if the folder is new</label>
          <input id="pgoal" value={goal} onChange={(e) => setGoal(e.target.value)} placeholder="predict churn" className={field} />
        </div>
        <Button type="submit" kind="primary" disabled={busy || !path.trim()}>Open project</Button>
      </form>
    </main>
  )
}
