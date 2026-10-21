// The dependency graph: steps in topological columns, real edges, status in the mark and the edge colour.
import { useMemo } from 'react'
import type { Step } from '../api'
import { STATUS_COLOR } from './ui'

const W = 150
const H = 34
const COL_GAP = 52
const ROW_GAP = 12
const PAD = 8

interface Node {
  step: Step
  col: number
  row: number
  x: number
  y: number
}

export function layered(steps: Step[]): Node[] {
  const byId = new Map(steps.map((s) => [s.id, s]))
  const depth = new Map<string, number>()
  const visit = (id: string, seen: Set<string>): number => {
    const known = depth.get(id)
    if (known != null) return known
    const deps = (byId.get(id)?.depends_on ?? []).filter((d) => byId.has(d) && !seen.has(d))
    const d = deps.length ? 1 + Math.max(...deps.map((x) => visit(x, new Set([...seen, id])))) : 0
    depth.set(id, d)
    return d
  }
  steps.forEach((s) => visit(s.id, new Set()))
  const rows = new Map<number, number>()
  return steps.map((step) => {
    const col = depth.get(step.id) ?? 0
    const row = rows.get(col) ?? 0
    rows.set(col, row + 1)
    return { step, col, row, x: PAD + col * (W + COL_GAP), y: PAD + row * (H + ROW_GAP) }
  })
}

function label(id: string): string {
  return id.length > 17 ? `${id.slice(0, 16)}…` : id
}

export function DepGraph({ steps, onOpen, active }: { steps: Step[]; onOpen: (id: string) => void; active?: string }) {
  const nodes = useMemo(() => layered(steps), [steps])
  if (!nodes.length) return null
  const cols = Math.max(...nodes.map((n) => n.col)) + 1
  const rows = Math.max(...nodes.map((n) => n.row)) + 1
  const width = PAD * 2 + cols * W + (cols - 1) * COL_GAP
  const height = PAD * 2 + rows * H + (rows - 1) * ROW_GAP
  const pos = new Map(nodes.map((n) => [n.step.id, n]))
  const edges = nodes.flatMap((n) =>
    n.step.depends_on.map((d) => ({ from: pos.get(d), to: n })).filter((e): e is { from: Node; to: Node } => !!e.from),
  )
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      className="h-auto max-w-full"
      preserveAspectRatio="xMinYMin meet"
      role="img"
      aria-label="dependency graph"
    >
      {edges.map(({ from, to }) => {
        const x1 = from.x + W
        const y1 = from.y + H / 2
        const x2 = to.x
        const y2 = to.y + H / 2
        const c = (x2 - x1) / 2
        const live = to.step.status === 'running'
        return (
          <path
            key={`${from.step.id}-${to.step.id}`}
            d={`M ${x1} ${y1} C ${x1 + c} ${y1}, ${x2 - c} ${y2}, ${x2} ${y2}`}
            fill="none"
            stroke={live ? '#F97316' : '#3F3F46'}
            strokeWidth={live ? 1.5 : 1}
            opacity={to.step.status === 'skipped' ? 0.5 : 1}
          />
        )
      })}
      {nodes.map((n) => {
        const colour = STATUS_COLOR[n.step.status] ?? STATUS_COLOR.pending
        const isActive = n.step.id === active
        return (
          <g
            key={n.step.id}
            transform={`translate(${n.x} ${n.y})`}
            className="cursor-pointer"
            onClick={() => onOpen(n.step.id)}
            tabIndex={0}
            role="button"
            aria-label={`open agent ${n.step.id}`}
            onKeyDown={(e) => e.key === 'Enter' && onOpen(n.step.id)}
          >
            <rect
              width={W}
              height={H}
              rx={2}
              fill="#18181B"
              stroke={isActive ? '#F97316' : n.step.status === 'running' ? '#F97316' : '#27272A'}
              strokeWidth={isActive ? 1.5 : 1}
              strokeOpacity={!isActive && n.step.status === 'running' ? 0.6 : 1}
            />
            <circle cx={14} cy={H / 2} r={4.5} fill={n.step.status === 'done' ? colour : 'none'} stroke={colour} strokeWidth={1.5}
              strokeDasharray={n.step.status === 'skipped' ? '2 2' : undefined} />
            {n.step.status === 'running' && <path d={`M 14 ${H / 2 - 4.5} A 4.5 4.5 0 0 1 14 ${H / 2 + 4.5} Z`} fill={colour} />}
            {n.step.status === 'failed' && (
              <path d={`M 11.5 ${H / 2 - 2.5} l 5 5 m 0 -5 l -5 5`} stroke={colour} strokeWidth={1.5} />
            )}
            <text x={26} y={H / 2 + 4} className="font-mono" fontSize={11.5} fill="#E4E4E7">
              {label(n.step.id)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
