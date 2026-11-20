import { useEffect, useState } from 'react'
import { getHealth } from '../api/client'

export default function HealthBadge() {
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  const checkHealth = async () => {
    setLoading(true)
    setError(false)
    try {
      const data = await getHealth()
      setHealth(data)
    } catch {
      setError(true)
      setHealth(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center gap-2 bg-white/10 backdrop-blur-sm rounded-full px-3 py-1.5">
        <div className="w-2.5 h-2.5 rounded-full bg-yellow-400 animate-pulse" />
        <span className="text-xs text-white/80 font-medium">Checking API…</span>
      </div>
    )
  }

  if (error || !health) {
    return (
      <button
        onClick={checkHealth}
        title="Click to retry"
        className="flex items-center gap-2 bg-red-500/20 border border-red-400/30 backdrop-blur-sm rounded-full px-3 py-1.5 hover:bg-red-500/30 transition-colors"
      >
        <div className="w-2.5 h-2.5 rounded-full bg-red-400" />
        <span className="text-xs text-red-300 font-medium">API Offline</span>
      </button>
    )
  }

  const modelLoaded = health.model_loaded
  const statusOk = health.status === 'ok' || health.status === 'healthy'

  return (
    <div
      title={`Model: ${health.model_path || 'unknown'}\nClasses: ${(health.classes || []).join(', ')}`}
      className={`flex items-center gap-2 backdrop-blur-sm rounded-full px-3 py-1.5 cursor-default transition-colors ${
        statusOk && modelLoaded
          ? 'bg-green-500/20 border border-green-400/30'
          : 'bg-yellow-500/20 border border-yellow-400/30'
      }`}
    >
      <div
        className={`w-2.5 h-2.5 rounded-full ${
          statusOk && modelLoaded
            ? 'bg-green-400 animate-pulse-slow'
            : 'bg-yellow-400 animate-pulse'
        }`}
      />
      <span
        className={`text-xs font-medium ${
          statusOk && modelLoaded ? 'text-green-300' : 'text-yellow-300'
        }`}
      >
        {statusOk && modelLoaded ? 'Model Ready' : 'No Model'}
      </span>
    </div>
  )
}
