// One plan step as a dense row: status, id, skill, model, elapsed, key metric, then goal and outcome.
import clsx from 'clsx'
import type { Step } from '../api'
import { elapsed, fmt } from '../session'
import { Chip, StatusMark } from './ui'

export function StepCard({ step, defaultModel, now, onOpen, active }: {
  step: Step
  defaultModel: string
  now: number
  onOpen: (id: string) => void
  active?: boolean
}) {
  const running = step.status === 'running'
  return (
    <button
      type="button"
      onClick={() => onOpen(step.id)}
      className={clsx(
        'group block w-full border-b border-border-sub px-3 py-2 text-left transition-colors duration-150',
        'hover:bg-surface focus-visible:bg-surface',
        active && 'bg-surface',
      )}
      aria-label={`${step.id}: ${step.status}`}
    >
      <div className="flex items-center gap-2">
        <StatusMark status={step.status} size={12} />
        <span className="font-mono text-sm font-medium text-fg">{step.id}</span>
        <span className="truncate text-xs text-muted">{step.skill}</span>
        {step.attempts > 1 && <Chip tone="muted">attempt {step.attempts}</Chip>}
        <span className="ml-auto flex shrink-0 items-center gap-3 font-mono text-xs tabular-nums">
          {step.metric && (
            <span>
              <span className="text-muted">{step.metric.name} </span>
              <span className="text-fg">{fmt(step.metric.value)}</span>
            </span>
          )}
          <span className={clsx(running ? 'text-orange' : 'text-muted')}>{elapsed(step.started, step.finished, now)}</span>
        </span>
      </div>
      <div className="mt-1 flex items-start gap-2 pl-5 text-xs">
        <p className="min-w-0 flex-1 truncate text-muted group-hover:text-fg/80">{step.goal}</p>
        <span className="hidden shrink-0 font-mono text-2xs text-muted sm:inline">{step.model ?? defaultModel}</span>
      </div>
      {(step.depends_on.length > 0 || step.last_failure || step.summary) && (
        <div className="mt-1 flex flex-wrap items-center gap-1.5 pl-5">
          {step.depends_on.map((d) => (
            <Chip key={d}>← {d}</Chip>
          ))}
          {step.status === 'failed' && step.last_failure ? (
            <span className="min-w-0 truncate text-xs text-danger">{step.last_failure}</span>
          ) : step.status === 'done' && step.summary ? (
            <span className="min-w-0 truncate text-xs text-muted">{step.summary}</span>
          ) : step.status === 'pending' && step.last_failure ? (
            <span className="min-w-0 truncate text-xs text-orange">retrying: {step.last_failure}</span>
          ) : null}
        </div>
      )}
    </button>
  )
}
