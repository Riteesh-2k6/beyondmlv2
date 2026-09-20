import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import clsx from 'clsx'
import Banner from '../components/Banner'

// ── Types ─────────────────────────────────────────────────────────────────────

type PathChoice =
  | 'autonomous'
  | 'explore'
  | 'supervised'
  | 'unsupervised'
  | 'dimensionality_reduction'
  | 'deep_learning'

type LLMChoice = 'ollama' | 'groq'

interface FormState {
  // Step 1: LLM
  llmChoice: LLMChoice
  groqApiKey: string
  groqModel: string
  ollamaModel: string
  // Step 2: Dataset
  file: File | null
  description: string
  pathChoice: PathChoice
  gaPop: number
  gaGen: number
  testSize: number
  useEnsemble: boolean
  targetCol: string
}

const PATH_OPTIONS: { id: PathChoice; label: string; desc: string }[] = [
  { id: 'autonomous',             label: 'Autonomous',          desc: 'Let the AI decide' },
  { id: 'explore',                label: 'Explore',             desc: 'EDA only' },
  { id: 'supervised',             label: 'Supervised',          desc: 'Classification / Regression' },
  { id: 'unsupervised',           label: 'Unsupervised',        desc: 'Clustering' },
  { id: 'dimensionality_reduction', label: 'Dim. Reduction',   desc: 'PCA / UMAP' },
  { id: 'deep_learning',          label: 'Deep Learning',       desc: 'Neural net (MLP)' },
]

// ── Sub-components ────────────────────────────────────────────────────────────

function Field({
  label, children,
}: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <label className="block font-mono text-2xs text-muted uppercase tracking-widest">
        {label}
      </label>
      {children}
    </div>
  )
}

function TextInput({
  value, onChange, placeholder, type = 'text', className = '',
}: {
  value: string
  onChange: (v: string) => void
  placeholder?: string
  type?: string
  className?: string
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className={clsx(
        'w-full bg-surface border border-border-sub rounded-sm px-3 py-2',
        'font-mono text-xs text-fg placeholder:text-muted',
        'focus:border-orange focus:outline-none transition-colors',
        className,
      )}
    />
  )
}

function NumInput({
  value, onChange, min, max, step = 1,
}: {
  value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(Number(e.target.value))}
      className={clsx(
        'w-full bg-surface border border-border-sub rounded-sm px-3 py-2',
        'font-mono text-xs text-fg',
        'focus:border-orange focus:outline-none transition-colors',
      )}
    />
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function SetupPage() {
  const navigate = useNavigate()
  const fileRef  = useRef<HTMLInputElement>(null)

  const [step, setStep] = useState<1 | 2>(1)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)

  const [form, setForm] = useState<FormState>({
    llmChoice: 'ollama',
    groqApiKey: '',
    groqModel: 'llama-3.3-70b-versatile',
    ollamaModel: 'qwen3:8b',
    file: null,
    description: '',
    pathChoice: 'autonomous',
    gaPop: 10,
    gaGen: 5,
    testSize: 0.20,
    useEnsemble: false,
    targetCol: '',
  })

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setForm((prev) => ({ ...prev, [k]: v }))

  // ── Step 1: LLM config ──────────────────────────────────────────────────────

  const step1Valid = form.llmChoice === 'ollama' || form.groqApiKey.trim().length > 0

  // ── Step 2: Upload + start ──────────────────────────────────────────────────

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) set('file', f)
  }

  const startPipeline = async () => {
    if (!form.file) return
    setLoading(true)
    setError(null)

    try {
      // 1. Upload file
      const fd = new FormData()
      fd.append('file', form.file)
      const upRes = await fetch('/api/upload', { method: 'POST', body: fd })
      if (!upRes.ok) throw new Error(`Upload failed: ${upRes.statusText}`)
      const upData = await upRes.json()
      const sid: string = upData.session_id

      // 2. Post config
      const cfgRes = await fetch(`/api/config/${sid}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          llm_choice:   form.llmChoice,
          groq_api_key: form.groqApiKey,
          groq_model:   form.groqModel,
          ollama_model: form.ollamaModel,
          path_choice:  form.pathChoice,
          ga_pop:       form.gaPop,
          ga_gen:       form.gaGen,
          test_size:    form.testSize,
          use_ensemble: form.useEnsemble,
          description:  form.description,
          target_col:   form.targetCol,
        }),
      })
      if (!cfgRes.ok) throw new Error(`Config failed: ${cfgRes.statusText}`)

      navigate(`/pipeline?session=${sid}`)
    } catch (e) {
      setError(String((e as Error).message))
      setLoading(false)
    }
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-obsidian flex flex-col items-center justify-start py-10 px-4">
      {/* Banner */}
      <div className="mb-6 overflow-x-auto w-full flex justify-center">
        <Banner />
      </div>
      <p className="font-mono text-2xs text-muted mb-8">
        Terminal-native AutoML · Ollama / Groq · Genetic Algorithm
      </p>

      {/* Step indicator */}
      <div className="flex items-center gap-3 mb-8">
        {[1, 2].map((s) => (
          <div key={s} className="flex items-center gap-3">
            <div
              className={clsx(
                'flex items-center justify-center w-6 h-6 rounded-sm border font-mono text-xs font-bold transition-colors',
                step === s
                  ? 'border-orange text-orange'
                  : s < step
                  ? 'border-success text-success'
                  : 'border-border-sub text-muted',
              )}
            >
              {s < step ? '✓' : s}
            </div>
            <span className={clsx('font-mono text-2xs', step === s ? 'text-fg' : 'text-muted')}>
              {s === 1 ? 'LLM Config' : 'Dataset'}
            </span>
            {s < 2 && <div className="w-8 h-px bg-border-sub" />}
          </div>
        ))}
      </div>

      {/* Card */}
      <div className="w-full max-w-[560px] bg-surface border border-border-sub rounded-sm p-6 space-y-5">

        {/* ── STEP 1 ── */}
        {step === 1 && (
          <>
            {/* LLM selector */}
            <Field label="LLM Provider">
              <div className="grid grid-cols-2 gap-2">
                {(['ollama', 'groq'] as LLMChoice[]).map((p) => (
                  <button
                    key={p}
                    onClick={() => set('llmChoice', p)}
                    className={clsx(
                      'py-2 px-3 border rounded-sm font-mono text-xs transition-colors text-left',
                      form.llmChoice === p
                        ? 'border-orange text-orange bg-orange/5'
                        : 'border-border-sub text-muted hover:border-muted',
                    )}
                  >
                    {p === 'ollama' ? 'Ollama (Local)' : 'Groq (Cloud)'}
                  </button>
                ))}
              </div>
            </Field>

            {form.llmChoice === 'groq' ? (
              <>
                <Field label="Groq API Key">
                  <TextInput
                    type="password"
                    value={form.groqApiKey}
                    onChange={(v) => set('groqApiKey', v)}
                    placeholder="gsk_..."
                  />
                </Field>
                <Field label="Groq Model">
                  <TextInput
                    value={form.groqModel}
                    onChange={(v) => set('groqModel', v)}
                    placeholder="llama-3.3-70b-versatile"
                  />
                </Field>
              </>
            ) : (
              <Field label="Ollama Model">
                <TextInput
                  value={form.ollamaModel}
                  onChange={(v) => set('ollamaModel', v)}
                  placeholder="qwen3:8b"
                />
              </Field>
            )}

            <button
              onClick={() => setStep(2)}
              disabled={!step1Valid}
              className={clsx(
                'w-full py-2 font-mono text-xs font-bold rounded-sm transition-colors mt-2',
                step1Valid
                  ? 'bg-orange text-obsidian hover:bg-orange/90'
                  : 'bg-border-sub text-muted cursor-not-allowed',
              )}
            >
              Next
            </button>
          </>
        )}

        {/* ── STEP 2 ── */}
        {step === 2 && (
          <>
            {/* File drop zone */}
            <Field label="Dataset">
              <div
                onClick={() => fileRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                className={clsx(
                  'border border-dashed rounded-sm px-4 py-5 cursor-pointer text-center transition-colors',
                  dragOver
                    ? 'border-orange bg-orange/5'
                    : form.file
                    ? 'border-success/50 bg-success/5'
                    : 'border-border-sub hover:border-muted',
                )}
              >
                {form.file ? (
                  <p className="font-mono text-xs text-success">{form.file.name}</p>
                ) : (
                  <p className="font-mono text-xs text-muted">
                    drop CSV / XLSX / JSON or click to browse
                  </p>
                )}
              </div>
              <input
                ref={fileRef}
                type="file"
                accept=".csv,.xlsx,.xls,.json"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && set('file', e.target.files[0])}
              />
            </Field>

            {/* Description */}
            <Field label="Description (optional)">
              <TextInput
                value={form.description}
                onChange={(v) => set('description', v)}
                placeholder="Brief description of this dataset..."
              />
            </Field>

            {/* Target col override */}
            <Field label="Target Column (optional — leave blank for auto)">
              <TextInput
                value={form.targetCol}
                onChange={(v) => set('targetCol', v)}
                placeholder="e.g. label, target, price"
              />
            </Field>

            {/* Path choice */}
            <Field label="Analysis Path">
              <div className="grid grid-cols-3 gap-2">
                {PATH_OPTIONS.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => set('pathChoice', p.id)}
                    className={clsx(
                      'py-2 px-2 border rounded-sm font-mono text-left transition-colors',
                      form.pathChoice === p.id
                        ? 'border-orange text-orange bg-orange/5'
                        : 'border-border-sub text-muted hover:border-muted',
                    )}
                  >
                    <p className="text-2xs font-bold">{p.label}</p>
                    <p className="text-[0.55rem] opacity-70 mt-0.5">{p.desc}</p>
                  </button>
                ))}
              </div>
            </Field>

            {/* GA / training params */}
            <div className="grid grid-cols-3 gap-3">
              <Field label="GA Pop">
                <NumInput value={form.gaPop} onChange={(v) => set('gaPop', v)} min={2} max={200} />
              </Field>
              <Field label="GA Gen">
                <NumInput value={form.gaGen} onChange={(v) => set('gaGen', v)} min={1} max={100} />
              </Field>
              <Field label="Test Split">
                <NumInput
                  value={form.testSize}
                  onChange={(v) => set('testSize', v)}
                  min={0.05} max={0.5} step={0.05}
                />
              </Field>
            </div>

            {/* Ensemble toggle */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => set('useEnsemble', !form.useEnsemble)}
                className={clsx(
                  'w-9 h-5 rounded-full border transition-colors relative',
                  form.useEnsemble
                    ? 'bg-orange border-orange'
                    : 'bg-surface border-border-sub',
                )}
              >
                <span
                  className={clsx(
                    'absolute top-0.5 w-4 h-4 rounded-full bg-fg transition-all',
                    form.useEnsemble ? 'left-4' : 'left-0.5',
                  )}
                />
              </button>
              <span className="font-mono text-2xs text-muted">
                Ensemble stacking
              </span>
            </div>

            {/* Error */}
            {error && (
              <p className="font-mono text-2xs text-danger border border-danger/30 rounded-sm px-3 py-2">
                {error}
              </p>
            )}

            {/* Actions */}
            <div className="flex gap-3 pt-1">
              <button
                onClick={() => setStep(1)}
                className="px-4 py-2 font-mono text-xs text-muted border border-border-sub rounded-sm hover:text-fg hover:border-muted transition-colors"
              >
                Back
              </button>
              <button
                onClick={startPipeline}
                disabled={!form.file || loading}
                className={clsx(
                  'flex-1 py-2 font-mono text-xs font-bold rounded-sm transition-colors',
                  form.file && !loading
                    ? 'bg-orange text-obsidian hover:bg-orange/90'
                    : 'bg-border-sub text-muted cursor-not-allowed',
                )}
              >
                {loading
                  ? 'Loading...'
                  : form.file
                  ? `Start Pipeline — ${form.file.name}`
                  : 'Select a dataset to continue'}
              </button>
            </div>
          </>
        )}
      </div>

      {/* Footer */}
      <p className="font-mono text-2xs text-muted/40 mt-8">
        All inference runs locally · no data leaves your machine
      </p>
    </div>
  )
}
