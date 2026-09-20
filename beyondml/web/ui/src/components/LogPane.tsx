import { useEffect, useRef, useState } from 'react'
import Plot from 'react-plotly.js'
import type { LogEntry } from '../hooks/usePipelineState'
import LogLine from './LogLine'

interface Props {
  entries: LogEntry[]
}

// Inject obsidian dark theme into every Plotly figure layout
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function applyDarkTheme(plotlyJson: any) {
  return {
    ...plotlyJson,
    layout: {
      ...(plotlyJson.layout ?? {}),
      paper_bgcolor: '#09090B',
      plot_bgcolor:  '#18181B',
      font: { color: '#E4E4E7', family: 'Geist Mono, monospace', size: 11 },
      xaxis: { ...(plotlyJson.layout?.xaxis ?? {}), gridcolor: '#27272A', zerolinecolor: '#27272A' },
      yaxis: { ...(plotlyJson.layout?.yaxis ?? {}), gridcolor: '#27272A', zerolinecolor: '#27272A' },
      margin: { l: 40, r: 16, t: 32, b: 40 },
      autosize: true,
    },
  }
}

function ChartEntry({ title, plotlyJson }: { title: string; plotlyJson: unknown }) {
  const themed = applyDarkTheme(plotlyJson)
  return (
    <div className="my-3 border border-border-sub rounded-sm overflow-hidden bg-surface">
      <p className="px-3 py-1 text-2xs font-mono text-muted border-b border-border-sub">
        {title}
      </p>
      <div className="plotly-chart-wrap w-full">
        <Plot
          data={themed.data ?? []}
          layout={{ ...themed.layout, height: 260 }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  )
}

export default function LogPane({ entries }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const paneRef   = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  // Auto-scroll to bottom unless user has scrolled up
  useEffect(() => {
    if (autoScroll) {
      bottomRef.current?.scrollIntoView({ behavior: 'instant' })
    }
  }, [entries, autoScroll])

  const handleScroll = () => {
    const el = paneRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40
    setAutoScroll(atBottom)
  }

  return (
    <div
      ref={paneRef}
      onScroll={handleScroll}
      className="flex-1 overflow-y-auto px-3 py-2 font-mono text-xs text-fg no-scrollbar"
    >
      {entries.map((entry) =>
        entry.isChart ? (
          <ChartEntry key={entry.id} title={entry.title} plotlyJson={entry.plotlyJson} />
        ) : (
          <LogLine key={entry.id} spans={entry.spans} timestamp={entry.timestamp} />
        )
      )}
      <div ref={bottomRef} />
    </div>
  )
}
