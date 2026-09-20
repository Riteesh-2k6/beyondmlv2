import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import SetupPage from './pages/SetupPage'
import PipelinePage from './pages/PipelinePage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"         element={<SetupPage />} />
        <Route path="/pipeline" element={<PipelinePage />} />
        <Route path="*"         element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
