import clsx from 'clsx'
import { useEffect, useRef, useState } from 'react'
import { BrowserRouter, Navigate, NavLink, Outlet, Route, Routes, useNavigate, useParams } from 'react-router-dom'
import { api } from './api'
import { ApprovalModal } from './components/ApprovalModal'
import { ChatDock } from './components/ChatDock'
import { Button, Key, StatusMark } from './components/ui'
import { SessionContext } from './context'
import { Agent } from './screens/Agent'
import { Experiments } from './screens/Experiments'
import { Files } from './screens/Files'
import { PlanBoard } from './screens/PlanBoard'
import { Projects } from './screens/Projects'
import { useSession } from './session'

const NAV = [
  { to: '', label: 'Plan', key: '1', end: true },
  { to: 'agents', label: 'Agents', key: '2', end: false },
  { to: 'experiments', label: 'Experiments', key: '3', end: false },
  { to: 'files', label: 'Files', key: '4', end: false },
]

function Shell() {
  const { name = '' } = useParams()
  const navigate = useNavigate()
  const session = useSession(name)
  const { summary, pending, connected } = session
  const chatInput = useRef<HTMLInputElement>(null)
  const [promptOpen, setPromptOpen] = useState(false)
  const lastPrompt = useRef<string | null>(null)
  const [stopping, setStopping] = useState(false)

  const first = pending[0]
  useEffect(() => {
    if (first && first.id !== lastPrompt.current) {
      lastPrompt.current = first.id
      setPromptOpen(true)
    }
  }, [first])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null
      const typing = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable)
      if (e.key === 'Escape' && typing) { (target as HTMLElement).blur(); return }
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return
      const nav = NAV.find((n) => n.key === e.key)
      if (nav) navigate(`/p/${encodeURIComponent(name)}/${nav.to}`)
      else if (e.key === 'c') { e.preventDefault(); chatInput.current?.focus() }
      else if (e.key === 'a' && pending.length) setPromptOpen(true)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [name, navigate, pending.length])

  async function answer(id: string, value: string | boolean) {
    await api.answer(name, id, value)
    session.refreshSummary()
  }

  async function stop() {
    setStopping(true)
    try { await api.stop(name) } finally { setStopping(false); session.refreshSummary() }
  }

  return (
    <SessionContext value={session}>
      <div className="flex h-full flex-col">
        <header className="flex h-10 shrink-0 items-center gap-3 border-b border-border-sub bg-surface px-3">
          <NavLink to="/" className="hidden text-sm font-medium text-orange hover:text-orange/80 sm:block" title="all projects">BeyondML</NavLink>
          <span className="shrink-0 whitespace-nowrap font-mono text-sm text-fg">{name}</span>
          {summary?.goal && <span className="hidden min-w-0 truncate text-sm text-muted md:block" style={{ maxWidth: '28rem' }}>{summary.goal}</span>}
          <nav aria-label="screens" className="flex min-w-0 items-center gap-0.5 overflow-x-auto sm:ml-2">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={`/p/${encodeURIComponent(name)}/${n.to}`}
                end={n.end}
                className={({ isActive }) => clsx('flex h-7 shrink-0 items-center gap-1.5 rounded-[2px] px-2 text-xs transition-colors duration-150',
                  isActive ? 'bg-obsidian text-orange' : 'text-muted hover:text-fg')}
              >
                {n.label}<span className="hidden sm:inline"><Key>{n.key}</Key></span>
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-3">
            {summary && <span className="hidden font-mono text-xs text-muted lg:block">{summary.mode} · {summary.model}</span>}
            <span className="flex items-center gap-1.5 font-mono text-xs" title={connected ? 'live event stream connected' : 'event stream reconnecting'}>
              <StatusMark status={summary?.running ? 'running' : connected ? 'done' : 'failed'} size={10} />
              <span className={summary?.running ? 'text-orange' : 'text-muted'}>{summary?.running ? 'running' : connected ? 'idle' : 'reconnecting'}</span>
            </span>
            {pending.length > 0 && (
              <button type="button" onClick={() => setPromptOpen(true)} className="flex h-7 items-center gap-1.5 rounded-[2px] border border-warn/60 px-2 text-xs text-warn hover:bg-warn/10">
                {pending.length} waiting <Key>a</Key>
              </button>
            )}
            {summary?.running && <Button kind="danger" onClick={() => void stop()} disabled={stopping}>Stop</Button>}
          </div>
        </header>
        <main className="min-h-0 flex-1 overflow-hidden">
          <Outlet />
        </main>
        <ChatDock project={name} messages={session.fold.chat} pendingCount={pending.length} onOpenPrompts={() => setPromptOpen(true)} inputRef={chatInput} />
        {promptOpen && first && <ApprovalModal prompt={first} count={pending.length} onAnswer={answer} onLater={() => setPromptOpen(false)} />}
      </div>
    </SessionContext>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Projects />} />
        <Route path="/p/:name" element={<Shell />}>
          <Route index element={<PlanBoard />} />
          <Route path="agents/:id?" element={<Agent />} />
          <Route path="experiments" element={<Experiments />} />
          <Route path="files/*" element={<Files />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
