// The evolve step's own view: best and mean fitness per generation, diversity, the best genome.
import type { Data } from 'plotly.js'
import { fmt, type Generation, type Trial } from '../session'
import { Plot, SERIES } from './Plot'

function ParamTable({ params }: { params: Record<string, unknown> }) {
  const rows = Object.entries(params)
  if (!rows.length) return <p className="text-xs text-muted">no parameters reported</p>
  return (
    <table className="font-mono text-xs">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k} className="border-b border-border-sub/60">
            <td className="py-1 pr-6 text-muted">{k}</td>
            <td className="py-1 text-fg tabular-nums">{fmt(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <span className="font-mono text-xs">
      <span className="text-muted">{label} </span>
      <span className={accent ? 'text-orange' : 'text-fg'}>{value}</span>
    </span>
  )
}

export function EvolveView({ generations, trials }: { generations: Generation[]; trials: Trial[] }) {
  const last = generations[generations.length - 1]
  const lastTrial = trials[trials.length - 1]
  return (
    <div className="space-y-6 p-3">
      {generations.length > 0 && last && (
        <section className="space-y-3">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
            <Stat label="generation" value={String(last.gen)} />
            <Stat label="evals" value={String(last.evals)} />
            <Stat label="elapsed" value={`${fmt(last.seconds)}s`} />
            <Stat label="best fitness" value={fmt(last.best_fitness)} accent />
            <Stat label="diversity" value={fmt(last.diversity)} />
          </div>
          <Plot
            height={260}
            data={[
              { x: generations.map((g) => g.gen), y: generations.map((g) => g.best_fitness), name: 'best', type: 'scatter', mode: 'lines+markers', line: { color: SERIES[0], width: 2 }, marker: { size: 5 } },
              { x: generations.map((g) => g.gen), y: generations.map((g) => g.mean_fitness), name: 'mean', type: 'scatter', mode: 'lines+markers', line: { color: '#71717A', width: 1.5 }, marker: { size: 4 } },
            ] as Data[]}
            layout={{ title: { text: 'fitness per generation', font: { size: 12, color: '#E4E4E7' }, x: 0 }, xaxis: { title: { text: 'generation' }, dtick: 1 } }}
          />
          <Plot
            height={160}
            data={[{ x: generations.map((g) => g.gen), y: generations.map((g) => g.diversity), type: 'bar', marker: { color: 'rgba(34,211,238,0.45)', line: { color: SERIES[1], width: 1 } }, name: 'diversity' }] as Data[]}
            layout={{ title: { text: 'population diversity', font: { size: 12, color: '#E4E4E7' }, x: 0 }, xaxis: { dtick: 1 }, yaxis: { range: [0, 1.05] }, showlegend: false }}
          />
          <div>
            <h3 className="mb-2 text-xs font-medium text-fg">best genome</h3>
            <ParamTable params={last.best_params ?? {}} />
          </div>
        </section>
      )}
      {trials.length > 0 && lastTrial && (
        <section className="space-y-3">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
            <Stat label="trial" value={String(lastTrial.trial)} />
            <Stat label="best value" value={fmt(lastTrial.best_value)} accent />
            <Stat label="elapsed" value={`${fmt(lastTrial.seconds)}s`} />
          </div>
          <Plot
            height={260}
            data={[
              { x: trials.map((t) => t.trial), y: trials.map((t) => t.value), name: 'value', type: 'scatter', mode: 'markers', marker: { color: SERIES[1], size: 5 } },
              { x: trials.map((t) => t.trial), y: trials.map((t) => t.best_value), name: 'best', type: 'scatter', mode: 'lines', line: { color: SERIES[0], width: 2 } },
            ] as Data[]}
            layout={{ title: { text: 'objective per trial', font: { size: 12, color: '#E4E4E7' }, x: 0 }, xaxis: { title: { text: 'trial' } } }}
          />
          <div>
            <h3 className="mb-2 text-xs font-medium text-fg">best parameters</h3>
            <ParamTable params={lastTrial.params ?? {}} />
          </div>
        </section>
      )}
    </div>
  )
}
