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

// One orange, then neutrals: the world has one accent and the charts keep to it.
export const SERIES = ['#F97316', '#A1A1AA', '#71717A', '#E4E4E7', '#D4D4D8', '#52525B']
export const NEUTRAL_FILL = 'rgba(161,161,170,0.35)'

const BASE_LAYOUT: Partial<Layout> = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: '"Geist Mono", ui-monospace, monospace', size: 11, color: '#A1A1AA' },
  margin: { l: 44, r: 12, t: 28, b: 36 },
  xaxis: { showgrid: false, zerolinecolor: '#27272A', linecolor: '#3F3F46', tickcolor: '#3F3F46' },
  yaxis: { gridcolor: '#27272A', zerolinecolor: '#27272A', linecolor: '#3F3F46', tickcolor: '#3F3F46' },
  legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 10 } },
  hoverlabel: { bgcolor: '#18181B', bordercolor: '#3F3F46', font: { family: '"Geist Mono", monospace', size: 11 } },
}

export function Plot({ data, layout, height = 260 }: { data: Data[]; layout?: Partial<Layout>; height?: number }) {
  // axes merge, so a caller's dtick or title keeps the palette's grid and line colours
  const xaxis = { ...BASE_LAYOUT.xaxis, ...layout?.xaxis }
  const xTitle = xaxis.title
  if (xTitle && typeof xTitle === 'object') xaxis.title = { standoff: 8, ...xTitle }
  const merged: Partial<Layout> = {
    ...BASE_LAYOUT,
    ...layout,
    xaxis,
    yaxis: { ...BASE_LAYOUT.yaxis, ...layout?.yaxis },
    margin: { ...BASE_LAYOUT.margin, ...layout?.margin, ...(xTitle ? { b: 48 } : {}) },
    height,
    autosize: true,
  }
  return (
    <Suspense fallback={<div className="skeleton" style={{ height }} aria-busy="true" />}>
      <LazyPlot
        data={data}
        layout={merged}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: '100%', height }}
      />
    </Suspense>
  )
}
