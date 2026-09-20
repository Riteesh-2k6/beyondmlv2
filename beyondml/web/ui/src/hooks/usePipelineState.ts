// All live pipeline state in one reducer — no prop drilling, no context boilerplate.

export interface Span { text: string; cls: string }

export interface LogLine {
  id: number
  spans: Span[]
  timestamp: string
  isChart?: false
}

export interface ChartLine {
  id: number
  isChart: true
  title: string
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  plotlyJson: any
}

export type LogEntry = LogLine | ChartLine

export interface TreeNode {
  name: string
  status: 'waiting' | 'running' | 'done' | 'error'
  children?: string[]
}

export interface PipelineStats {
  shape?: string
  target?: string
  task?: string
  classes?: string
  split?: string
  new_features?: string
  test_score?: string
  train_score?: string
  fitness_latest?: string
}

export interface PipelineState {
  logEntries: LogEntry[]
  stats: PipelineStats
  treeNodes: Record<string, TreeNode>
  gaFitness: number[]
  inputRequest: string | null   // prompt string when waiting for user input
  done: boolean
  evalResult: Record<string, unknown> | null
  error: string | null
  connected: boolean
}

const INITIAL_TREE_NODES: Record<string, TreeNode> = {
  'Orchestrator':              { name: 'Orchestrator',              status: 'waiting' },
  'Domain Sanity (Parallel)':  { name: 'Domain Sanity (Parallel)',  status: 'waiting' },
  'Data Drift Analysis':       { name: 'Data Drift Analysis',       status: 'waiting' },
  'Imputation':                { name: 'Imputation',                status: 'waiting' },
  'EDA Agent (Parallel)':      { name: 'EDA Agent (Parallel)',      status: 'waiting' },
  'Outlier Handler (Parallel)':{ name: 'Outlier Handler (Parallel)',status: 'waiting' },
  'Feature Engineer':          { name: 'Feature Engineer',          status: 'waiting' },
  'GA Trainer (Supervised)':   { name: 'GA Trainer (Supervised)',   status: 'waiting' },
  'Ensemble':                  { name: 'Ensemble',                  status: 'waiting' },
  'DL Trainer (Neural Net)':   { name: 'DL Trainer (Neural Net)',   status: 'waiting' },
  'Evaluator':                 { name: 'Evaluator',                 status: 'waiting' },
  'Explainability':            { name: 'Explainability',            status: 'waiting' },
  'Reflection':                { name: 'Reflection',                status: 'waiting' },
}

export const INITIAL_STATE: PipelineState = {
  logEntries: [],
  stats: {},
  treeNodes: INITIAL_TREE_NODES,
  gaFitness: [],
  inputRequest: null,
  done: false,
  evalResult: null,
  error: null,
  connected: false,
}

let _nextId = 0

export type PipelineAction =
  | { type: 'CONNECTED' }
  | { type: 'DISCONNECTED' }
  | { type: 'LOG';        spans: Span[];  timestamp: string }
  | { type: 'CHART';      title: string; plotlyJson: unknown }
  | { type: 'STAT';       key: string;   value: string }
  | { type: 'TREE';       node: string;  status: TreeNode['status']; children?: string[] }
  | { type: 'GA_PROGRESS'; fitness: number[] }
  | { type: 'INPUT_REQUEST'; prompt: string }
  | { type: 'INPUT_CLEAR' }
  | { type: 'DONE';       evalResult: Record<string, unknown> }
  | { type: 'ERROR';      message: string }

export function pipelineReducer(
  state: PipelineState,
  action: PipelineAction,
): PipelineState {
  switch (action.type) {
    case 'CONNECTED':
      return { ...state, connected: true, error: null }

    case 'DISCONNECTED':
      return { ...state, connected: false }

    case 'LOG':
      return {
        ...state,
        logEntries: [
          ...state.logEntries,
          { id: _nextId++, spans: action.spans, timestamp: action.timestamp },
        ],
      }

    case 'CHART':
      return {
        ...state,
        logEntries: [
          ...state.logEntries,
          { id: _nextId++, isChart: true, title: action.title, plotlyJson: action.plotlyJson },
        ],
      }

    case 'STAT':
      return {
        ...state,
        stats: { ...state.stats, [action.key]: action.value },
      }

    case 'TREE': {
      const existing = state.treeNodes[action.node] ?? { name: action.node }
      return {
        ...state,
        treeNodes: {
          ...state.treeNodes,
          [action.node]: {
            ...existing,
            name: action.node,
            status: action.status,
            ...(action.children !== undefined ? { children: action.children } : {}),
          },
        },
      }
    }

    case 'GA_PROGRESS':
      return { ...state, gaFitness: action.fitness }

    case 'INPUT_REQUEST':
      return { ...state, inputRequest: action.prompt }

    case 'INPUT_CLEAR':
      return { ...state, inputRequest: null }

    case 'DONE':
      return { ...state, done: true, evalResult: action.evalResult, inputRequest: null }

    case 'ERROR':
      return { ...state, error: action.message, inputRequest: null }

    default:
      return state
  }
}
