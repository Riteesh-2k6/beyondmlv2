import { useState, useRef, useEffect } from 'react'
import clsx from 'clsx'

interface Props {
  inputRequest: string | null
  onSubmit: (value: string) => void
  connected: boolean
  done: boolean
}

export default function InputBar({ inputRequest, onSubmit, connected, done }: Props) {
  const [value, setValue] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (inputRequest) {
      inputRef.current?.focus()
      setValue('')
    }
  }, [inputRequest])

  const handleKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      onSubmit(value)
      setValue('')
    }
  }

  const statusDot = connected
    ? done
      ? 'bg-success'
      : 'bg-orange animate-ping-slow'
    : 'bg-muted'

  return (
    <div className="flex items-center gap-3 px-3 py-2 bg-surface border-t border-border-sub min-h-[42px]">
      {/* Connection status dot */}
      <span className="relative flex h-2 w-2 shrink-0">
        <span className={clsx('relative inline-flex rounded-full h-2 w-2', statusDot)} />
      </span>

      {inputRequest ? (
        <>
          <span className="font-mono text-2xs text-orange shrink-0 max-w-[45%] truncate">
            {inputRequest}
          </span>
          <input
            ref={inputRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Type response and press Enter..."
            className="flex-1 bg-transparent font-mono text-xs text-fg placeholder:text-muted outline-none border-b border-orange pb-[1px]"
          />
        </>
      ) : (
        <span className="font-mono text-2xs text-muted">
          {done ? 'pipeline complete' : connected ? 'running...' : 'connecting...'}
          {!done && <span className="mx-3 opacity-40">·</span>}
          {!done && <span className="opacity-40">c charts</span>}
        </span>
      )}
    </div>
  )
}
