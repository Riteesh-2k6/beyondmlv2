import { AreaChart, Area, ResponsiveContainer, Tooltip } from 'recharts'
import type { PipelineStats } from '../hooks/usePipelineState'

interface Props {
  stats: PipelineStats
  gaFitness: number[]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  columns?: string[]
  previewRows?: string[][]
}

function StatRow({ label, value }: { label: string; value?: string }) {
  if (!value) return null
  return (
    <div className="flex justify-between gap-2 font-mono text-2xs py-[2px]">
      <span className="text-muted">{label}</span>
      <span className="text-fg font-bold truncate max-w-[120px] text-right">{value}</span>
    </div>
  )
}

export default function StatsPanel({ stats, gaFitness, columns, previewRows }: Props) {
  const fitnessData = gaFitness.map((v, i) => ({ i, v }))

  return (
    <div className="flex flex-col h-full overflow-y-auto no-scrollbar text-xs">
      {/* ── Dataset stats ── */}
      <div className="px-3 pt-3 pb-2 border-b border-border-sub">
        <p className="font-mono text-2xs text-muted uppercase tracking-widest mb-1">Dataset</p>
        <StatRow label="shape"    value={stats.shape} />
        <StatRow label="target"   value={stats.target} />
        <StatRow label="task"     value={stats.task} />
        <StatRow label="classes"  value={stats.classes} />
        <StatRow label="split"    value={stats.split} />
        <StatRow label="features" value={stats.new_features} />
      </div>

      {/* ── Model scores ── */}
      {(stats.test_score || stats.train_score) && (
        <div className="px-3 pt-2 pb-2 border-b border-border-sub">
          <p className="font-mono text-2xs text-muted uppercase tracking-widest mb-1">Score</p>
          <StatRow label="test"  value={stats.test_score} />
          <StatRow label="train" value={stats.train_score} />
        </div>
      )}

      {/* ── GA Fitness sparkline ── */}
      <div className="px-3 pt-2 pb-2 border-b border-border-sub">
        <p className="font-mono text-2xs text-muted uppercase tracking-widest mb-1">GA Fitness</p>
        {fitnessData.length > 1 ? (
          <ResponsiveContainer width="100%" height={48}>
            <AreaChart data={fitnessData} margin={{ top: 2, right: 0, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="fitGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#F97316" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#F97316" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <Tooltip
                contentStyle={{ background: '#18181B', border: '1px solid #27272A', fontSize: 10, color: '#E4E4E7' }}
                formatter={(v) => (typeof v === 'number' ? v.toFixed(2) : String(v ?? ''))}
                labelFormatter={() => ''}
              />
              <Area
                type="monotone"
                dataKey="v"
                stroke="#F97316"
                strokeWidth={1.5}
                fill="url(#fitGrad)"
                dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-muted text-2xs font-mono">waiting for GA...</p>
        )}
      </div>

      {/* ── Data preview table ── */}
      {columns && previewRows && (
        <div className="px-3 pt-2 pb-3 flex-1 overflow-hidden">
          <p className="font-mono text-2xs text-muted uppercase tracking-widest mb-1">Data Preview</p>
          <div className="overflow-x-auto overflow-y-auto max-h-52 no-scrollbar">
            <table className="text-2xs font-mono border-collapse w-full min-w-max">
              <thead>
                <tr>
                  {columns.map((c) => (
                    <th
                      key={c}
                      className="text-left px-2 py-[2px] text-muted border-b border-border-sub sticky top-0 bg-obsidian whitespace-nowrap"
                    >
                      {c.slice(0, 12)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {previewRows.map((row, ri) => (
                  <tr key={ri} className={ri % 2 === 0 ? 'bg-obsidian' : 'bg-surface'}>
                    {row.map((cell, ci) => (
                      <td key={ci} className="px-2 py-[1px] text-fg/80 whitespace-nowrap">
                        {String(cell).slice(0, 12)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
