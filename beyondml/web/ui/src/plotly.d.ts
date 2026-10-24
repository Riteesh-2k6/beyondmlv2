declare module 'plotly.js/dist/plotly-basic.min.js' {
  import type { PlotlyHTMLElement } from 'plotly.js'
  const Plotly: typeof import('plotly.js') & { newPlot: (...args: unknown[]) => Promise<PlotlyHTMLElement> }
  export default Plotly
}
