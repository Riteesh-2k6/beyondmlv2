import { useEffect, useRef, useCallback, useReducer } from 'react'
import { pipelineReducer, INITIAL_STATE } from './usePipelineState'
import type { PipelineState } from './usePipelineState'

export function usePipelineWS(sessionId: string | null): {
  state: PipelineState
  sendUserResponse: (value: string) => void
} {
  const [state, dispatch] = useReducer(pipelineReducer, INITIAL_STATE)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!sessionId) return

    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host  = import.meta.env.DEV ? 'localhost:7777' : window.location.host
    const url   = `${proto}://${host}/ws/${sessionId}`

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => dispatch({ type: 'CONNECTED' })

    ws.onclose = () => dispatch({ type: 'DISCONNECTED' })

    ws.onerror = () =>
      dispatch({ type: 'ERROR', message: 'WebSocket connection failed' })

    ws.onmessage = (ev) => {
      let msg: { type: string; payload: Record<string, unknown> }
      try {
        msg = JSON.parse(ev.data)
      } catch {
        return
      }

      const { type, payload } = msg

      switch (type) {
        case 'log':
          dispatch({
            type: 'LOG',
            spans: (payload.spans as { text: string; cls: string }[]) ?? [],
            timestamp: (payload.timestamp as string) ?? '',
          })
          break

        case 'stat':
          dispatch({
            type: 'STAT',
            key:   payload.key   as string,
            value: payload.value as string,
          })
          break

        case 'tree_update':
          dispatch({
            type:     'TREE',
            node:     payload.node   as string,
            status:   payload.status as 'waiting' | 'running' | 'done' | 'error',
            children: payload.children as string[] | undefined,
          })
          break

        case 'chart':
          dispatch({
            type:       'CHART',
            title:      payload.title      as string,
            plotlyJson: payload.plotly_json,
          })
          break

        case 'ga_progress':
          dispatch({
            type:    'GA_PROGRESS',
            fitness: payload.fitness as number[],
          })
          break

        case 'input_request':
          dispatch({ type: 'INPUT_REQUEST', prompt: payload.prompt as string })
          break

        case 'done':
          dispatch({
            type:       'DONE',
            evalResult: payload as Record<string, unknown>,
          })
          break

        case 'error':
          dispatch({ type: 'ERROR', message: payload.message as string })
          break
      }
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [sessionId])

  const sendUserResponse = useCallback((value: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ type: 'user_response', payload: { value } })
      )
      dispatch({ type: 'INPUT_CLEAR' })
    }
  }, [])

  return { state, sendUserResponse }
}
