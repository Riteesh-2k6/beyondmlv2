// The lead agent's dock: the transcript and one input, pinned to the bottom of every screen.
import clsx from 'clsx'
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { api } from '../api'
import type { ChatMessage } from '../session'
import { Button, Key, field } from './ui'

const KEY = 'beyondml.chat.collapsed'

export function ChatDock({ project, messages, pendingCount, onOpenPrompts, inputRef }: {
  project: string
  messages: ChatMessage[]
  pendingCount: number
  onOpenPrompts: () => void
  inputRef: React.RefObject<HTMLInputElement | null>
}) {
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem(KEY) === '1')
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const log = useRef<HTMLDivElement>(null)

  useEffect(() => {
    localStorage.setItem(KEY, collapsed ? '1' : '0')
  }, [collapsed])

  useEffect(() => {
    const el = log.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages.length, busy, collapsed])

  async function send(e: FormEvent) {
    e.preventDefault()
    const text = draft.trim()
    if (!text || busy) return
    setDraft('')
    setBusy(true)
    setError(null)
    try {
      await api.chat(project, text)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <section
      aria-label="chat with the lead agent"
      className={clsx('flex shrink-0 flex-col border-t border-border-sub bg-obsidian', collapsed ? 'h-9' : 'h-52')}
    >
      <div className="flex h-9 shrink-0 items-center gap-3 px-3">
        <button
          type="button"
          onClick={() => setCollapsed((c) => !c)}
          className="flex items-center gap-2 text-xs text-fg hover:text-orange"
          aria-expanded={!collapsed}
        >
          <svg width="10" height="10" viewBox="0 0 10 10" className={clsx('transition-transform duration-150', collapsed && '-rotate-90')}>
            <path d="M2 3.5 L5 6.5 L8 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
          lead agent
        </button>
        {busy && <span className="text-xs text-orange">thinking…</span>}
        {pendingCount > 0 && (
          <button type="button" onClick={onOpenPrompts} className="text-xs text-warn hover:underline">
            {pendingCount} prompt{pendingCount > 1 ? 's' : ''} waiting for you
          </button>
        )}
        <span className="ml-auto hidden items-center gap-1 text-2xs text-muted sm:flex">
          <Key>c</Key> focus <Key>esc</Key> leave
        </span>
      </div>
      {!collapsed && (
        <>
          <div ref={log} className="min-h-0 flex-1 overflow-y-auto px-3 pb-1 text-sm">
            {messages.length === 0 && (
              <p className="pt-1 text-xs text-muted">
                Ask the lead what the plan is doing, why a step failed, or tell it to change the plan. It reads the
                project, never runs code.
              </p>
            )}
            {messages.map((m, i) => (
              <div key={`${m.ts}-${i}`} className="flex gap-2 py-0.5 leading-5">
                <span className={clsx('w-9 shrink-0 font-mono text-xs leading-5', m.agent === 'user' ? 'text-info' : 'text-orange')}>
                  {m.agent === 'user' ? 'you' : 'lead'}
                </span>
                <p className="min-w-0 whitespace-pre-wrap break-words text-fg/90">{m.text}</p>
              </div>
            ))}
            {error && <p className="py-0.5 text-xs text-danger">lead error: {error}</p>}
          </div>
          <form onSubmit={send} className="flex shrink-0 items-center gap-2 border-t border-border-sub px-3 py-1.5">
            <input
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="message the lead agent"
              className={field}
              aria-label="message the lead agent"
              disabled={busy}
            />
            <Button type="submit" kind="primary" disabled={busy || !draft.trim()}>
              Send
            </Button>
          </form>
        </>
      )}
    </section>
  )
}
