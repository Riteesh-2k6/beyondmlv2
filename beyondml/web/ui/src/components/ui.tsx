// Small shared pieces: the status mark, mono chips, panel headings, empty states, buttons.
import clsx from 'clsx'
import type { ReactNode } from 'react'
import type { Status } from '../api'

export const STATUS_COLOR: Record<string, string> = {
  pending: '#71717A', running: '#F97316', done: '#22C55E', failed: '#EF4444', skipped: '#52525B', 'n/a': '#52525B',
}

/** A drawn status mark: ring, half, disc, cross, dotted ring. One stroke weight everywhere. */
export function StatusMark({ status, size = 12, className }: { status: Status | string; size?: number; className?: string }) {
  const c = STATUS_COLOR[status] ?? STATUS_COLOR.pending
  const r = size / 2 - 1.5
  const cx = size / 2
  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      className={clsx('shrink-0', status === 'running' && 'mark-running', className)}
      aria-label={status}
      role="img"
    >
      {status === 'done' && <circle cx={cx} cy={cx} r={r} fill={c} />}
      {status === 'pending' && <circle cx={cx} cy={cx} r={r} fill="none" stroke={c} strokeWidth={1.5} />}
      {status === 'skipped' && (
        <circle cx={cx} cy={cx} r={r} fill="none" stroke={c} strokeWidth={1.5} strokeDasharray="2 2" />
      )}
      {status === 'running' && (
        <>
          <circle cx={cx} cy={cx} r={r} fill="none" stroke={c} strokeWidth={1.5} />
          <path d={`M ${cx} ${cx - r} A ${r} ${r} 0 0 1 ${cx} ${cx + r} Z`} fill={c} />
        </>
      )}
      {status === 'failed' && (
        <>
          <circle cx={cx} cy={cx} r={r} fill="none" stroke={c} strokeWidth={1.5} />
          <path
            d={`M ${cx - r / 2} ${cx - r / 2} L ${cx + r / 2} ${cx + r / 2} M ${cx + r / 2} ${cx - r / 2} L ${cx - r / 2} ${cx + r / 2}`}
            stroke={c}
            strokeWidth={1.5}
          />
        </>
      )}
      {status === 'n/a' && <circle cx={cx} cy={cx} r={1.5} fill={c} />}
    </svg>
  )
}

export function StatusWord({ status }: { status: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-xs" style={{ color: STATUS_COLOR[status] }}>
      <StatusMark status={status} />
      {status}
    </span>
  )
}

export function Chip({ children, tone = 'muted', className }: { children: ReactNode; tone?: 'muted' | 'accent' | 'warn' | 'danger'; className?: string }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-[2px] border px-1.5 py-px font-mono text-2xs leading-4',
        tone === 'muted' && 'border-border-sub text-muted',
        tone === 'accent' && 'border-orange/40 text-orange',
        tone === 'warn' && 'border-warn/50 text-warn',
        tone === 'danger' && 'border-danger/50 text-danger',
        className,
      )}
    >
      {children}
    </span>
  )
}

export function PanelTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex h-8 shrink-0 items-center justify-between border-b border-border-sub bg-surface px-3">
      <h2 className="text-xs font-medium text-fg">{children}</h2>
      {right && <div className="flex items-center gap-2 text-2xs text-muted">{right}</div>}
    </div>
  )
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="m-4 max-w-prose border border-dashed border-border-sub p-4 text-sm">
      <p className="font-medium text-fg">{title}</p>
      {children && <div className="mt-1 text-muted">{children}</div>}
    </div>
  )
}

export function Button({
  children, onClick, kind = 'ghost', disabled, type = 'button', className, title,
}: {
  children: ReactNode
  onClick?: () => void
  kind?: 'primary' | 'ghost' | 'danger'
  disabled?: boolean
  type?: 'button' | 'submit'
  className?: string
  title?: string
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={clsx(
        'inline-flex h-7 items-center gap-1.5 rounded-[2px] border px-2.5 text-xs transition-colors duration-150',
        'disabled:cursor-not-allowed disabled:opacity-40',
        kind === 'primary' && 'border-orange bg-orange text-obsidian hover:bg-orange/90 disabled:hover:bg-orange',
        kind === 'ghost' && 'border-border-sub text-fg hover:border-muted hover:bg-surface',
        kind === 'danger' && 'border-danger/60 text-danger hover:bg-danger/10',
        className,
      )}
    >
      {children}
    </button>
  )
}

export const field =
  'h-7 w-full rounded-[2px] border border-border-sub bg-obsidian px-2 text-xs text-fg placeholder:text-muted ' +
  'focus:border-orange focus:outline-none'

export function Key({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded-[2px] border border-border-sub bg-surface px-1 font-mono text-2xs text-muted">{children}</kbd>
  )
}

export function timeOf(ts: number): string {
  const d = new Date(ts * 1000)
  return d.toTimeString().slice(0, 8)
}
