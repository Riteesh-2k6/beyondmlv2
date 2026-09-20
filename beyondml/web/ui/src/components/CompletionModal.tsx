interface Props {
  evalResult: Record<string, unknown> | null
  onExport: () => void
  onClose: () => void
}

export default function CompletionModal({ evalResult, onExport, onClose }: Props) {
  const score = evalResult?.test_score as number | undefined
  const model = evalResult?.model_type as string | undefined
  const importances = evalResult?.feature_importances as Record<string, number> | undefined

  const topFeats = importances
    ? Object.entries(importances)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
    : []

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-obsidian/80 backdrop-blur-sm">
      <div className="bg-surface border border-border-sub rounded-sm w-80 font-mono text-xs">
        {/* Header */}
        <div className="px-4 py-3 border-b border-border-sub">
          <p className="text-success font-bold">Pipeline Complete</p>
        </div>

        {/* Body */}
        <div className="px-4 py-3 space-y-1">
          <div className="flex justify-between">
            <span className="text-muted">model</span>
            <span className="text-orange font-bold">{model ?? '—'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted">test score</span>
            <span className="text-success font-bold">
              {score != null ? (score * 100).toFixed(2) + '%' : '—'}
            </span>
          </div>

          {topFeats.length > 0 && (
            <>
              <div className="border-t border-border-sub my-2" />
              <p className="text-muted mb-1">top features</p>
              {topFeats.map(([col, imp]) => (
                <div key={col} className="flex justify-between">
                  <span className="text-fg truncate max-w-[60%]">{col}</span>
                  <span className="text-muted">{imp.toFixed(4)}</span>
                </div>
              ))}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-border-sub flex gap-2 justify-end">
          <button
            onClick={onClose}
            className="px-3 py-1 text-xs text-muted border border-border-sub rounded-sm hover:text-fg hover:border-muted transition-colors"
          >
            close
          </button>
          <button
            onClick={onExport}
            className="px-3 py-1 text-xs bg-orange text-obsidian font-bold rounded-sm hover:bg-orange/90 transition-colors"
          >
            export notebook
          </button>
        </div>
      </div>
    </div>
  )
}
