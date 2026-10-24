// A `chart` event: a Plotly figure JSON under reports/charts/ (fetched through the files route),
// or an inline {kind, x, series} fallback.
import { useEffect, useState } from 'react'
import type { Data, Layout } from 'plotly.js'
import { api } from '../api'
import { useProject } from '../context'
import type { ChartData } from '../session'
import { Plot, SERIES } from './Plot'

interface Figure {
  data: Data[]
  layout?: Partial<Layout>
}

function inline(chart: ChartData): Data[] {
  const names = Object.keys(chart.series ?? {})
  const x = chart.x ?? []
  return names.map((name, i) => {
    const colour = SERIES[i % SERIES.length]
    const y = chart.series![name]
    if (chart.kind === 'bar') return { x, y, name, type: 'bar', marker: { color: colour } }
    if (chart.kind === 'scatter') return { x, y, name, type: 'scatter', mode: 'markers', marker: { color: colour, size: 5 } }
    return { x, y, name, type: 'scatter', mode: 'lines+markers', line: { color: colour, width: 1.5 }, marker: { size: 4 } }
  })
}

/** Fold indices, generations, trials: whole numbers get whole-number ticks. */
function integerAxis(xs: unknown[]): boolean {
  return xs.length > 0 && xs.length <= 30 && xs.every((v) => typeof v === 'number' && Number.isInteger(v))
}

/** Traces keep their own colours when the file sets them; unset ones take the palette in order. */
function recolour(data: Data[]): Data[] {
  return data.map((trace, i) => {
    const t = trace as Record<string, unknown>
    const marker = (t.marker as Record<string, unknown> | undefined) ?? {}
    const line = (t.line as Record<string, unknown> | undefined) ?? {}
    const colour = SERIES[i % SERIES.length]
    return {
      ...t,
      marker: { ...marker, color: marker.color ?? colour },
      line: { ...line, color: line.color ?? colour },
    } as Data
  })
}

export function ChartEvent({ chart }: { chart: ChartData }) {
  const session = useProject()
  const [figure, setFigure] = useState<Figure | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!chart.path) return
    let live = true
    setFigure(null)
    api.file(session.name, chart.path)
      .then(({ text }) => {
        const parsed = JSON.parse(text) as Figure
        if (!Array.isArray(parsed.data)) throw new Error('not a Plotly figure: no data array')
        if (live) setFigure(parsed)
      })
      .catch((e) => { if (live) setError(e instanceof Error ? e.message : String(e)) })
    return () => { live = false }
  }, [session.name, chart.path, session.diskVersion])

  const title = { text: chart.title, font: { size: 12, color: '#E4E4E7' }, x: 0 }
  if (chart.path) {
    if (error) return <p className="text-xs text-danger">{chart.title}: {chart.path} could not be read ({error})</p>
    if (!figure) return <div className="skeleton" style={{ height: 240 }} aria-busy="true" />
    const { height: _h, width: _w, template: _t, paper_bgcolor: _p, plot_bgcolor: _b, ...layout } = (figure.layout ?? {}) as Record<string, unknown>
    const xs = figure.data.flatMap((t) => ((t as { x?: unknown[] }).x ?? []))
    return <Plot data={recolour(figure.data)} height={260} layout={{ ...(layout as Partial<Layout>), title, ...(integerAxis(xs) ? { xaxis: { ...((layout as Partial<Layout>).xaxis ?? {}), dtick: 1 } } : {}) }} />
  }
  if (!chart.x?.length || !chart.series || !Object.keys(chart.series).length) {
    return <p className="text-xs text-muted">{chart.title}: nothing to plot</p>
  }
  return (
    <Plot data={inline(chart)} height={240}
      layout={{ title, showlegend: Object.keys(chart.series).length > 1, ...(integerAxis(chart.x) ? { xaxis: { dtick: 1 } } : {}) }} />
  )
}
