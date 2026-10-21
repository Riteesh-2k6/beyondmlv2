// An agent's live stream: messages, tool calls and results, status changes, prompts, errors; newest at the bottom.
import clsx from 'clsx'
import { useEffect, useRef, useState } from 'react'
import type { BusEvent } from '../api'
import { fmt } from '../session'
import { StatusWord, timeOf } from './ui'

function isFailure(resp: unknown): resp is Record<string, unknown> {
  return !!resp && typeof resp === 'object' && ((resp as Record<string, unknown>).ok === false || 'error' in (resp as object))
}

function Chevron() {
  return (
    <svg width="8" height="8" viewBox="0 0 8 8" className="chev mr-1 inline-block text-muted transition-transform duration-150" aria-hidden="true">
      <path d="M2.5 1.5 L5.5 4 L2.5 6.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  )
}

function Row({ ev }: { ev: BusEvent }) {
  const d = ev.data
  const time = <span className="w-16 shrink-0 font-mono text-2xs text-muted tabular-nums">{timeOf(ev.ts)}</span>
  switch (ev.kind) {
    case 'message_delta':
      return (
        <div className="flex gap-2 py-0.5">{time}<p className="min-w-0 whitespace-pre-wrap break-words text-fg/90">{String(d.text ?? '')}</p></div>
      )
    case 'tool_call':
      return (
        <div className="flex gap-2 py-0.5">
          {time}
          <details className="min-w-0 flex-1 font-mono text-xs">
            <summary className="cursor-pointer list-none text-orange">
              <Chevron />{String(d.name)}
              <span className="text-muted">({JSON.stringify(d.args ?? {}).slice(0, 140)})</span>
            </summary>
            <pre className="mt-1 max-h-72 overflow-auto whitespace-pre-wrap break-words border border-border-sub bg-obsidian p-2 text-muted">
              {JSON.stringify(d.args ?? {}, null, 2)}
            </pre>
          </details>
        </div>
      )
    case 'tool_result': {
      const failed = isFailure(d.response)
      const detail = failed ? String((d.response as Record<string, unknown>).error ?? (d.response as Record<string, unknown>).errors ?? '') : ''
      return (
        <div className="flex gap-2 py-0.5">
          {time}
          <details className="min-w-0 flex-1 font-mono text-xs">
            <summary className={clsx('cursor-pointer list-none', failed ? 'text-danger' : 'text-success')}>
              <Chevron />{String(d.name)}{failed ? <span className="text-muted">  {detail.slice(0, 160)}</span> : <span className="text-muted">  ok</span>}
            </summary>
            <pre className="mt-1 max-h-72 overflow-auto whitespace-pre-wrap break-words border border-border-sub bg-obsidian p-2 text-muted">
              {JSON.stringify(d.response ?? null, null, 2)}
            </pre>
          </details>
        </div>
      )
    }
    case 'step_status':
      return (
        <div className="flex items-center gap-2 py-1">{time}<StatusWord status={String(d.status)} />{d.reason ? <span className="text-xs text-muted">{String(d.reason)}</span> : null}</div>
      )
    case 'question':
      if (!('id' in d)) return null
      return (
        <div className="flex gap-2 py-0.5">{time}<p className="text-sm"><span className="text-orange">? </span>{String(d.question)}
          {d.cancelled ? <span className="text-muted"> → cancelled</span> : 'answer' in d && <span className="text-success"> → {String(d.answer)}</span>}</p></div>
      )
    case 'approval_request':
      if (!('id' in d)) return null
      return (
        <div className="flex gap-2 py-0.5">{time}<p className="text-sm"><span className="text-orange">! </span>approval: {String(d.tool ?? d.kind)}
          {d.cancelled ? <span className="text-muted"> → cancelled</span>
            : 'answer' in d && <span className={d.answer ? 'text-success' : 'text-danger'}> → {d.answer ? 'approved' : 'denied'}</span>}</p></div>
      )
    case 'generation':
      return (
        <div className="flex gap-2 py-0.5 font-mono text-xs text-muted">{time}<span>gen {String(d.gen)}  best {fmt(d.best_fitness)}  mean {fmt(d.mean_fitness)}  diversity {fmt(d.diversity)}</span></div>
      )
    case 'trial':
      return (
        <div className="flex gap-2 py-0.5 font-mono text-xs text-muted">{time}<span>trial {String(d.number)}  {String(d.state ?? '').toLowerCase()}  value {fmt(d.value)}</span></div>
      )
    case 'artifact':
      return <div className="flex gap-2 py-0.5 font-mono text-xs text-muted">{time}<span>wrote {String(d.path)}</span></div>
    case 'chart':
      return <div className="flex gap-2 py-0.5 font-mono text-xs text-muted">{time}<span>chart: {String(d.title)}{d.path ? ` (${String(d.path)})` : ''}</span></div>
    case 'budget':
      return <div className="flex gap-2 py-0.5 text-sm">{time}<span className="text-orange">budget: {String(d.reason)}</span></div>
    case 'error':
      return <div className="flex gap-2 py-0.5 text-sm">{time}<span className="text-danger">{String(d.code)}: {String(d.message)}</span></div>
    default:
      return <div className="flex gap-2 py-0.5 font-mono text-xs text-muted">{time}<span>{ev.kind} {JSON.stringify(d).slice(0, 160)}</span></div>
  }
}

export function LiveLog({ events }: { events: BusEvent[] }) {
  const box = useRef<HTMLDivElement>(null)
  const [stuck, setStuck] = useState(true)
  const [unseen, setUnseen] = useState(0)
  const lastCount = useRef(events.length)

  useEffect(() => {
    const el = box.current
    if (!el) return
    if (stuck) {
      el.scrollTop = el.scrollHeight
      setUnseen(0)
    } else if (events.length > lastCount.current) {
      setUnseen((n) => n + events.length - lastCount.current)
    }
    lastCount.current = events.length
  }, [events.length, stuck])

  function onScroll() {
    const el = box.current
    if (!el) return
    setStuck(el.scrollHeight - el.scrollTop - el.clientHeight < 24)
  }

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div ref={box} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto px-3 py-2 text-sm">
        {events.length === 0 && <p className="text-xs text-muted">Nothing yet. Tool calls and messages appear here the moment the agent starts.</p>}
        {events.map((ev, i) => <Row key={`${ev.ts}-${i}`} ev={ev} />)}
      </div>
      {!stuck && unseen > 0 && (
        <button
          type="button"
          onClick={() => { setStuck(true); const el = box.current; if (el) el.scrollTop = el.scrollHeight }}
          className="absolute bottom-3 right-4 rounded-[2px] border border-orange bg-surface px-2 py-1 font-mono text-xs text-orange"
        >
          ↓ {unseen} new
        </button>
      )}
    </div>
  )
}
