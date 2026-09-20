import type { Span } from '../hooks/usePipelineState'

interface Props {
  spans: Span[]
  timestamp?: string
}

// Map backend cls values to CSS classes defined in index.css
const CLS_MAP: Record<string, string> = {
  'agent-header': 'log-agent-header',
  'success':      'log-success',
  'info':         'log-info',
  'warn':         'log-warn',
  'error-text':   'log-error-text',
  'accent':       'log-accent',
  'muted':        'log-muted',
  'bold':         'log-bold',
  'italic':       'log-italic',
}

export default function LogLine({ spans, timestamp }: Props) {
  return (
    <p className="leading-5 whitespace-pre-wrap break-words min-h-[1.25rem]">
      {timestamp && (
        <span className="log-muted mr-2 text-2xs select-none">{timestamp}</span>
      )}
      {spans.map((span, i) => (
        <span key={i} className={CLS_MAP[span.cls] ?? ''}>
          {span.text}
        </span>
      ))}
    </p>
  )
}
