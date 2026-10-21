// Files: the project tree (lazy, hides .git) and a preview of the selected file.
import clsx from 'clsx'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api, type FileEntry } from '../api'
import { Empty, PanelTitle } from '../components/ui'
import { useProject } from '../context'

const HIDDEN = new Set(['.git', '.beyondml', '__pycache__'])
const MAX_PREVIEW = 200_000

function parseCsv(text: string, maxRows: number): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let cell = ''
  let quoted = false
  for (let i = 0; i < text.length && rows.length <= maxRows; i++) {
    const ch = text[i]
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++ } else if (ch === '"') quoted = false
      else cell += ch
    } else if (ch === '"') quoted = true
    else if (ch === ',') { row.push(cell); cell = '' }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++
      row.push(cell); rows.push(row); row = []; cell = ''
    } else cell += ch
  }
  if (cell || row.length) { row.push(cell); rows.push(row) }
  return rows
}

function Tree({ path, depth, selected, onSelect, listings, load, expanded, toggle }: {
  path: string; depth: number; selected: string; onSelect: (p: string) => void
  listings: Record<string, FileEntry[] | undefined>; load: (p: string) => void
  expanded: Set<string>; toggle: (p: string) => void
}) {
  const entries = listings[path]
  useEffect(() => { if (!entries) load(path) }, [entries, load, path])
  if (!entries) return <div className="skeleton mx-2 my-1 h-5" />
  return (
    <ul>
      {entries.filter((e) => !HIDDEN.has(e.name)).map((e) => {
        const full = path ? `${path}/${e.name}` : e.name
        const isOpen = expanded.has(full)
        return (
          <li key={full}>
            <button
              type="button"
              onClick={() => (e.dir ? toggle(full) : onSelect(full))}
              style={{ paddingLeft: 8 + depth * 12 }}
              className={clsx('flex w-full items-center gap-1.5 py-0.5 pr-2 text-left font-mono text-xs transition-colors duration-150 hover:bg-surface',
                selected === full ? 'bg-surface text-orange' : e.dir ? 'text-fg' : 'text-fg/80')}
              aria-expanded={e.dir ? isOpen : undefined}
              aria-current={selected === full ? 'true' : undefined}
            >
              {e.dir ? (
                <svg width="10" height="10" viewBox="0 0 10 10" className={clsx('shrink-0 text-muted transition-transform duration-150', isOpen && 'rotate-90')}>
                  <path d="M3.5 2 L6.5 5 L3.5 8" fill="none" stroke="currentColor" strokeWidth="1.5" />
                </svg>
              ) : <span className="w-2.5 shrink-0" />}
              <span className="truncate">{e.name}</span>
              {!e.dir && <span className="ml-auto shrink-0 text-2xs text-muted tabular-nums">{human(e.size)}</span>}
            </button>
            {e.dir && isOpen && (
              <Tree path={full} depth={depth + 1} selected={selected} onSelect={onSelect} listings={listings} load={load} expanded={expanded} toggle={toggle} />
            )}
          </li>
        )
      })}
      {entries.filter((e) => !HIDDEN.has(e.name)).length === 0 && <li className="px-3 py-0.5 text-xs text-muted" style={{ paddingLeft: 20 + depth * 12 }}>empty</li>}
    </ul>
  )
}

function human(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

type Preview = { kind: 'text'; text: string; truncated: boolean } | { kind: 'csv'; rows: string[][] } | { kind: 'image'; url: string } | { kind: 'binary'; size: number } | { kind: 'error'; message: string }

export function Files() {
  const session = useProject()
  const navigate = useNavigate()
  const params = useParams()
  const selected = params['*'] ?? ''
  const [listings, setListings] = useState<Record<string, FileEntry[] | undefined>>({})
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(['src', 'data', 'reports', 'agents']))
  const [preview, setPreview] = useState<Preview | null>(null)

  const load = useCallback((path: string) => {
    api.list(session.name, path)
      .then(({ entries }) => setListings((l) => ({ ...l, [path]: entries })))
      .catch(() => setListings((l) => ({ ...l, [path]: [] })))
  }, [session.name])

  useEffect(() => { setListings({}) }, [session.name, session.diskVersion])

  useEffect(() => {
    if (!selected) { setPreview(null); return }
    let live = true
    api.file(session.name, selected)
      .then(({ text, type, size }) => {
        if (!live) return
        if (type.startsWith('image/')) setPreview({ kind: 'image', url: api.fileUrl(session.name, selected) })
        else if (text.slice(0, 4096).includes('\0')) setPreview({ kind: 'binary', size })
        else if (/\.(csv|tsv)$/i.test(selected)) setPreview({ kind: 'csv', rows: parseCsv(text.slice(0, MAX_PREVIEW), 50) })
        else setPreview({ kind: 'text', text: text.slice(0, MAX_PREVIEW), truncated: size > MAX_PREVIEW })
      })
      .catch((e) => { if (live) setPreview({ kind: 'error', message: e instanceof Error ? e.message : String(e) }) })
    return () => { live = false }
  }, [session.name, selected, session.diskVersion])

  const select = (p: string) => navigate(`/p/${encodeURIComponent(session.name)}/files/${p}`)
  const toggle = (p: string) => setExpanded((s) => { const n = new Set(s); if (n.has(p)) n.delete(p); else n.add(p); return n })
  const lines = useMemo(() => (preview?.kind === 'text' ? preview.text.split('\n').slice(0, 3000) : []), [preview])

  return (
    <div className="flex h-full min-h-0 flex-col md:flex-row">
      <nav aria-label="project files" className="max-h-56 shrink-0 overflow-y-auto border-b border-border-sub py-1 md:max-h-none md:w-64 md:border-b-0 md:border-r">
        <Tree path="" depth={0} selected={selected} onSelect={select} listings={listings} load={load} expanded={expanded} toggle={toggle} />
      </nav>
      <section className="flex min-h-0 flex-1 flex-col" aria-label="file preview">
        <PanelTitle right={selected && <a href={api.fileUrl(session.name, selected)} target="_blank" rel="noreferrer" className="hover:text-fg">open raw</a>}>
          <span className="font-mono">{selected || 'select a file'}</span>
        </PanelTitle>
        <div className="min-h-0 flex-1 overflow-auto">
          {!selected && (
            <Empty title="The project is the repo you ship">
              src/ holds the agents' scripts, data/processed their intermediate files, models/ and reports/ the results.
              agents/&lt;id&gt;/ keeps each agent's exact instruction and event log.
            </Empty>
          )}
          {preview?.kind === 'error' && <p className="p-3 text-xs text-danger">{preview.message}</p>}
          {preview?.kind === 'binary' && <p className="p-3 text-xs text-muted">binary file, {human(preview.size)}</p>}
          {preview?.kind === 'image' && <img src={preview.url} alt={selected} className="max-w-full p-3" />}
          {preview?.kind === 'csv' && (
            <table className="m-0 border-collapse font-mono text-xs">
              <thead className="sticky top-0 bg-obsidian">
                <tr>{preview.rows[0]?.map((h, i) => <th key={i} className="whitespace-nowrap border-b border-border-sub px-2 py-1 text-left font-normal text-muted">{h}</th>)}</tr>
              </thead>
              <tbody>
                {preview.rows.slice(1, 51).map((r, i) => (
                  <tr key={i} className="border-b border-border-sub/50">{r.map((c, j) => <td key={j} className="whitespace-nowrap px-2 py-0.5 tabular-nums text-fg/90">{c}</td>)}</tr>
                ))}
              </tbody>
            </table>
          )}
          {preview?.kind === 'text' && (
            <>
              {preview.truncated && <p className="px-3 pt-2 text-2xs text-muted">showing the first 200 KB</p>}
              <pre className="px-3 py-2 font-mono text-xs leading-5 text-fg/90">
                {lines.map((l, i) => (
                  <span key={i} className="block whitespace-pre-wrap break-words">
                    <span className="mr-3 inline-block w-8 select-none text-right text-muted tabular-nums">{i + 1}</span>{l}
                  </span>
                ))}
              </pre>
            </>
          )}
        </div>
      </section>
    </div>
  )
}
