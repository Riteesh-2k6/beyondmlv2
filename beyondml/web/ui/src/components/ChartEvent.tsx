// A `chart` event rendered with Plotly: {title, kind, x, series}.
import type { Data } from 'plotly.js'
import type { ChartData } from '../session'
import { Plot, SERIES } from './Plot'

export function ChartEvent({ chart }: { chart: ChartData }) {
  const names = Object.keys(chart.series ?? {})
  if (!chart.x?.length || !names.length) {
    return <p className="text-xs text-muted">{chart.title}: nothing to plot</p>
  }
  const data: Data[] = names.map((name, i) => {
    const colour = SERIES[i % SERIES.length]
    const y = chart.series[name]
    if (chart.kind === 'bar') return { x: chart.x, y, name, type: 'bar', marker: { color: colour } }
    if (chart.kind === 'scatter') return { x: chart.x, y, name, type: 'scatter', mode: 'markers', marker: { color: colour, size: 5 } }
    return { x: chart.x, y, name, type: 'scatter', mode: 'lines+markers', line: { color: colour, width: 1.5 }, marker: { size: 4 } }
  })
  return (
    <Plot
      data={data}
      height={240}
      layout={{ title: { text: chart.title, font: { size: 12, color: '#E4E4E7' }, x: 0 }, showlegend: names.length > 1 }}
    />
  )
}
