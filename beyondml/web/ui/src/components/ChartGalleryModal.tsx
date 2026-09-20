import { useState } from 'react'
import Plot from 'react-plotly.js'
import type { ChartLine } from '../hooks/usePipelineState'

interface Props {
  charts: ChartLine[]
  onClose: () => void
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function applyDark(json: any) {
  return {
    ...json,
    layout: {
      ...(json.layout ?? {}),
      paper_bgcolor: '#09090B',
      plot_bgcolor:  '#18181B',
      font:  { color: '#E4E4E7', family: 'Geist Mono, monospace', size: 11 },
      xaxis: { ...(json.layout?.xaxis ?? {}), gridcolor: '#27272A', zerolinecolor: '#27272A' },
      yaxis: { ...(json.layout?.yaxis ?? {}), gridcolor: '#27272A', zerolinecolor: '#27272A' },
      margin: { l: 40, r: 16, t: 32, b: 40 },
      autosize: true,
    },
  }
}

export default function ChartGalleryModal({ charts, onClose }: Props) {
  const [active, setActive] = useState<number | null>(null)

  if (charts.length === 0) return null

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col bg-obsidian/95 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border-sub shrink-0">
        <span className="font-mono text-xs text-muted">
          Chart Gallery
          <span className="text-orange ml-2">{charts.length}</span>
        </span>
        <button
          onClick={onClose}
          className="font-mono text-xs text-muted hover:text-fg transition-colors px-2 py-1"
        >
          close [esc]
        </button>
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto p-4">
        {active !== null ? (
          // Full-screen single chart
          <div className="h-full flex flex-col">
            <button
              onClick={() => setActive(null)}
              className="font-mono text-2xs text-muted hover:text-fg mb-2 self-start"
            >
              ← back to grid
            </button>
            <div className="flex-1 border border-border-sub rounded-sm overflow-hidden bg-surface">
              <p className="px-3 py-1 text-2xs font-mono text-muted border-b border-border-sub">
                {charts[active].title}
              </p>
              <Plot
                data={applyDark(charts[active].plotlyJson).data ?? []}
                layout={{ ...applyDark(charts[active].plotlyJson).layout, height: undefined, autosize: true }}
                config={{ displayModeBar: true, responsive: true }}
                style={{ width: '100%', height: 'calc(100% - 30px)' }}
                useResizeHandler
              />
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {charts.map((chart, i) => {
              const themed = applyDark(chart.plotlyJson)
              return (
                <div
                  key={chart.id}
                  onClick={() => setActive(i)}
                  className="border border-border-sub rounded-sm overflow-hidden bg-surface cursor-pointer hover:border-orange transition-colors"
                >
                  <p className="px-3 py-1 text-2xs font-mono text-muted border-b border-border-sub">
                    {chart.title}
                  </p>
                  <Plot
                    data={themed.data ?? []}
                    layout={{ ...themed.layout, height: 220 }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                    useResizeHandler
                  />
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
