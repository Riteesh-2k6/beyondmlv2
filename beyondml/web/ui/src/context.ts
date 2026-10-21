import { createContext, useContext } from 'react'
import type { Session } from './session'

export const SessionContext = createContext<Session | null>(null)

export function useProject(): Session {
  const s = useContext(SessionContext)
  if (!s) throw new Error('useProject outside a project route')
  return s
}
