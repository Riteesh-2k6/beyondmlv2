import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { usePipelineWS } from '../hooks/usePipelineWS'
import type { ChartLine } from '../hooks/usePipelineState'
import PipelineTree from '../components/PipelineTree'
import LogPane from '../components/LogPane'
import StatsPanel from '../components/StatsPanel'
import InputBar from '../components/InputBar'
import ChartGalleryModal from '../components/ChartGalleryModal'
import CompletionModal from '../components/CompletionModal'

export default function PipelinePage() {
  const [params]   = useSearchParams()
  const navigate   = useNavigate()
  const sessionId  = params.get('session')

  const { state, sendUserResponse } = usePipelineWS(sessionId)

  const [showGallery,    setShowGallery]    = useState(false)
  const [showCompletion, setShowCompletion] = useState(false)
  const [previewCols,    setPreviewCols]    = useState<string[]>([])

  // Redirect back if no session
  useEffect(() => {
    if (!sessionId) navigate('/', { replace: true })
  }, [sessionId, navigate])

  // Show completion modal when done arrives
  useEffect(() => {
    if (state.done) setShowCompletion(true)
  }, [state.done])

  // Fetch dataset columns / preview once on mount
  useEffect(() => {
    if (!sessionId) return
    fetch(`/api/session/${sessionId}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.columns) setPreviewCols(data.columns.slice(0, 20))
      })
      .catch(() => {})
  }, [sessionId])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'c' && !state.inputRequest) setShowGallery((v) => !v)
      if (e.key === 'Escape') { setShowGallery(false); setShowCompletion(false) }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [state.inputRequest])

  // Collect chart entries for gallery
  const charts = state.logEntries.filter((e): e is ChartLine => !!e.isChart)

  if (!sessionId) return null

  return (
    <div className="flex flex-col h-screen bg-obsidian overflow-hidden">

      {/* ── 3-column main area ────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left: Pipeline tree */}
        <div className="flex flex-col w-[200px] shrink-0 panel-side overflow-hidden">
          <div className="px-3 py-1.5 border-b border-border-sub bg-surface shrink-0">
            <span className="font-mono text-2xs text-muted uppercase tracking-widest">
              Pipeline
            </span>
          </div>
          <div className="flex-1 overflow-y-auto no-scrollbar">
            <PipelineTree nodes={state.treeNodes} />
          </div>
        </div>

        {/* Center: Live log + inline charts */}
        <div className="flex flex-col flex-1 panel-center overflow-hidden">
          <div className="px-3 py-1.5 border-b border-orange/40 bg-surface shrink-0 flex items-center justify-between">
            <span className="font-mono text-2xs text-muted uppercase tracking-widest">
              Agent Log
            </span>
            {charts.length > 0 && (
              <button
                onClick={() => setShowGallery(true)}
                className="font-mono text-2xs text-orange hover:text-orange/80 transition-colors"
              >
                {charts.length} chart{charts.length > 1 ? 's' : ''} [c]
              </button>
            )}
          </div>
          <LogPane entries={state.logEntries} />
        </div>

        {/* Right: Stats + data preview */}
        <div className="flex flex-col w-[220px] shrink-0 panel-side overflow-hidden">
          <div className="px-3 py-1.5 border-b border-border-sub bg-surface shrink-0">
            <span className="font-mono text-2xs text-muted uppercase tracking-widest">
              Stats
            </span>
          </div>
          <div className="flex-1 overflow-hidden">
            <StatsPanel
              stats={state.stats}
              gaFitness={state.gaFitness}
              columns={previewCols}
              previewRows={undefined}
            />
          </div>
        </div>
      </div>

      {/* ── Bottom input bar ──────────────────────────────────── */}
      <InputBar
        inputRequest={state.inputRequest}
        onSubmit={sendUserResponse}
        connected={state.connected}
        done={state.done}
      />

      {/* ── Overlays ──────────────────────────────────────────── */}
      {showGallery && (
        <ChartGalleryModal
          charts={charts}
          onClose={() => setShowGallery(false)}
        />
      )}

      {showCompletion && (
        <CompletionModal
          evalResult={state.evalResult}
          onClose={() => setShowCompletion(false)}
          onExport={() => {
            // Export fires a POST to /api/export/{session_id} — stubbed for now
            fetch(`/api/export/${sessionId}`, { method: 'POST' })
              .then(() => setShowCompletion(false))
              .catch(() => setShowCompletion(false))
          }}
        />
      )}

      {/* Connection error banner */}
      {state.error && !state.done && (
        <div className="absolute bottom-12 left-0 right-0 mx-4 py-2 px-4 bg-danger/10 border border-danger/30 rounded-sm font-mono text-xs text-danger">
          {state.error}
        </div>
      )}
    </div>
  )
}
