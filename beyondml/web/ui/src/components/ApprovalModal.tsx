import clsx from 'clsx'
// A pending prompt, answered here: plan approval, tool approval, budget, or a free-text question.
import { useEffect, useRef, useState, type FormEvent } from 'react'
import type { Prompt } from '../api'
import { Button, Key, field } from './ui'

const TITLE: Record<string, string> = {
  plan: 'approve the plan', tool: 'allow a tool call', budget: 'budget reached', question: 'question',
}

export function ApprovalModal({ prompt, count, onAnswer, onLater }: {
  prompt: Prompt
  count: number
  onAnswer: (id: string, value: string | boolean) => Promise<void>
  onLater: () => void
}) {
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const isQuestion = prompt.event === 'question'
  const kind = isQuestion ? 'question' : (prompt.kind ?? 'tool')

  useEffect(() => {
    setText('')
    setError(null)
    if (isQuestion && !(prompt.options ?? []).length) input.current?.focus() // with choices, digit keys pick; tab reaches the input
  }, [prompt.id, isQuestion])

  const options = isQuestion ? (prompt.options ?? []).filter((o) => typeof o === 'string' && o.trim()) : []

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onLater()
      if (busy || (e.target as HTMLElement)?.tagName === 'INPUT') return
      if (isQuestion) {
        const n = Number(e.key)
        if (n >= 1 && n <= Math.min(options.length, 9)) void answer(options[n - 1])
        return
      }
      if (e.key === 'y') void answer(true)
      if (e.key === 'n') void answer(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  async function answer(value: string | boolean) {
    setBusy(true)
    setError(null)
    try {
      await onAnswer(prompt.id, value)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault()
    if (text.trim()) void answer(text.trim())
  }

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-obsidian/70 p-4" role="presentation">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="prompt-title"
        className="modal-in w-full max-w-2xl border border-orange bg-surface shadow-[0_12px_40px_-8px_rgba(0,0,0,0.8)]"
      >
        <div className="flex items-center gap-2 border-b border-border-sub px-4 py-2.5">
          <span className="font-mono text-sm text-orange">{prompt.agent}</span>
          <span className="text-muted">·</span>
          <h2 id="prompt-title" className="text-sm font-medium text-fg">{TITLE[kind] ?? kind}</h2>
          {count > 1 && <span className="ml-auto text-xs text-muted">{count - 1} more after this</span>}
        </div>
        <div className="max-h-[50vh] overflow-y-auto px-4 py-3 text-sm">
          {kind === 'plan' && (
            <ol className="space-y-1.5">
              {(prompt.plan?.steps ?? []).map((s) => (
                <li key={s.id} className="flex gap-2">
                  <span className="w-28 shrink-0 font-mono text-xs text-fg">{s.id}</span>
                  <span className="w-28 shrink-0 truncate font-mono text-xs text-muted">{s.skill}</span>
                  <span className="min-w-0 text-fg/90">{s.goal}</span>
                </li>
              ))}
              {!prompt.plan?.steps?.length && <li className="text-muted">The plan has no steps.</li>}
            </ol>
          )}
          {kind === 'tool' && (
            <>
              <p className="font-mono text-sm text-fg">{prompt.tool}</p>
              <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-words border border-border-sub bg-obsidian p-2 font-mono text-xs text-muted">
                {JSON.stringify(prompt.args ?? {}, null, 1)}
              </pre>
              {prompt.hint && <p className="mt-2 text-xs text-muted">{prompt.hint}</p>}
            </>
          )}
          {kind === 'budget' && <p className="text-fg/90">{prompt.reason}. Approving grants one more budget's worth.</p>}
          {isQuestion && <p className="whitespace-pre-wrap text-fg/90">{prompt.question}</p>}
          {options.length > 0 && (
            <ol className="mt-3 space-y-1" aria-label="choices">
              {options.map((o, i) => {
                const rec = o === prompt.recommended
                return (
                  <li key={o}>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void answer(o)}
                      className={clsx('flex w-full items-center gap-2 rounded-[2px] border px-2 py-1.5 text-left text-sm transition-colors duration-150',
                        rec ? 'border-orange/60 text-fg hover:bg-orange/10' : 'border-border-sub text-fg/90 hover:border-muted hover:bg-obsidian')}
                    >
                      {i < 9 && <Key>{i + 1}</Key>}
                      <span className="min-w-0 flex-1">{o}</span>
                      {rec && <span className="font-mono text-2xs text-orange">recommended</span>}
                    </button>
                  </li>
                )
              })}
            </ol>
          )}
          {error && <p className="mt-2 text-xs text-danger">{error}</p>}
        </div>
        <div className="flex items-center gap-2 border-t border-border-sub px-4 py-2.5">
          {isQuestion ? (
            <form onSubmit={submit} className="flex w-full items-center gap-2">
              <input ref={input} value={text} onChange={(e) => setText(e.target.value)} className={field}
                placeholder={options.length ? 'or type your own answer' : 'your answer'} aria-label="your answer" disabled={busy} />
              <Button type="submit" kind="primary" disabled={busy || !text.trim()}>Answer</Button>
              <Button onClick={onLater}>Later</Button>
            </form>
          ) : (
            <>
              <Button kind="primary" onClick={() => void answer(true)} disabled={busy}>
                Approve <Key>y</Key>
              </Button>
              <Button kind="danger" onClick={() => void answer(false)} disabled={busy}>
                Deny <Key>n</Key>
              </Button>
              <Button onClick={onLater} className="ml-auto">
                Later <Key>esc</Key>
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
