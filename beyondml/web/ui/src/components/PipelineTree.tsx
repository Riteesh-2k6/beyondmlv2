import type { TreeNode } from '../hooks/usePipelineState'
import clsx from 'clsx'

interface Props {
  nodes: Record<string, TreeNode>
}

const NODE_ORDER = [
  'Orchestrator',
  'Domain Sanity (Parallel)',
  'Data Drift Analysis',
  'Imputation',
  'EDA Agent (Parallel)',
  'Outlier Handler (Parallel)',
  'Feature Engineer',
  'GA Trainer (Supervised)',
  'Ensemble',
  'DL Trainer (Neural Net)',
  'Evaluator',
  'Explainability',
  'Reflection',
]

function StatusDot({ status }: { status: TreeNode['status'] }) {
  return (
    <span className="relative flex h-2 w-2 shrink-0">
      {status === 'running' && (
        <span className="animate-ping-slow absolute inline-flex h-full w-full rounded-full bg-orange opacity-75" />
      )}
      <span
        className={clsx('relative inline-flex rounded-full h-2 w-2', {
          'bg-muted':   status === 'waiting',
          'bg-orange':  status === 'running',
          'bg-success': status === 'done',
          'bg-danger':  status === 'error',
        })}
      />
    </span>
  )
}

export default function PipelineTree({ nodes }: Props) {
  return (
    <div className="py-1">
      {NODE_ORDER.map((name) => {
        const node = nodes[name] ?? { name, status: 'waiting' as const }
        return (
          <div
            key={name}
            className={clsx(
              'flex items-center gap-2 px-3 py-[3px] font-mono text-2xs transition-colors',
              node.status === 'running' ? 'text-fg' : 'text-muted',
              node.status === 'done'    && 'text-success/80',
              node.status === 'error'   && 'text-danger',
            )}
          >
            <StatusDot status={node.status} />
            <span className="truncate">{name}</span>
          </div>
        )
      })}
    </div>
  )
}
