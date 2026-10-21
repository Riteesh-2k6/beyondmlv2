// Plotly in the obsidian palette, loaded on demand so the main bundle stays small.
import { lazy, Suspense } from 'react'
import type { Data, Layout } from 'plotly.js'

const LazyPlot = lazy(async () => {
  const [{ default: Plotly }, { default: createPlotlyComponent }] = await Promise.all([
    import('plotly.js/dist/plotly-basic.min.js'),
    import('react-plotly.js/factory'),
  ])
  return { default: createPlotlyComponent(Plotly) }
})

export const SERIES = ['#F97316', '#22D3EE', '#D946EF', '#22C55E', '#EAB308', '#A1A1AA']

const BASE_LAYOUT: Partial<Layout> = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: '"Geist Mono", ui-monospace, monospace', size: 11, color: '#A1A1AA' },
  margin: { l: 44, r: 12, t: 28, b: 36 },
  xaxis: { gridcolor: '#27272A', zerolinecolor: '#27272A', linecolor: '#3F3F46', tickcolor: '#3F3F46' },
  yaxis: { gridcolor: '#27272A', zerolinecolor: '#27272A', linecolor: '#3F3F46', tickcolor: '#3F3F46' },
  legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 10 } },
  hoverlabel: { bgcolor: '#18181B', bordercolor: '#3F3F46', font: { family: '"Geist Mono", monospace', size: 11 } },
}

export function Plot({ data, layout, height = 260 }: { data: Data[]; layout?: Partial<Layout>; height?: number }) {
  return (
    <Suspense fallback={<div className="skeleton" style={{ height }} aria-busy="true" />}>
      <LazyPlot
        data={data}
        layout={{ ...BASE_LAYOUT, ...layout, height, autosize: true }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: '100%', height }}
      />
    </Suspense>
  )
}
