import { useState, useCallback } from 'react'
import { Zap, RefreshCw, AlertCircle, Layers, ScanSearch } from 'lucide-react'
import HealthBadge from './components/HealthBadge'
import UploadZone from './components/UploadZone'
import DetectionResult from './components/DetectionResult'
import BatchResults from './components/BatchResults'
import { detectDefects, getAnnotatedImage } from './api/client'

// ─── Stats Bar ───────────────────────────────────────────────────────────────
function StatsBar({ predictions, inferenceTime }) {
  const total = predictions.length
  const defects = predictions.filter((p) => p.class_name?.toLowerCase() === 'defect').length
  const insulators = predictions.filter((p) => p.class_name?.toLowerCase() === 'insulator').length

  const stats = [
    { label: 'Total Detections', value: total, color: 'text-gray-800', bg: 'bg-gray-50' },
    { label: 'Defects', value: defects, color: 'text-red-600', bg: 'bg-red-50' },
    { label: 'Insulators', value: insulators, color: 'text-green-600', bg: 'bg-green-50' },
    {
      label: 'Inference Time',
      value: inferenceTime !== undefined ? `${inferenceTime.toFixed(1)} ms` : '—',
      color: 'text-blue-600',
      bg: 'bg-blue-50',
    },
  ]

  return (
    <div className="grid grid-cols-4 gap-3">
      {stats.map(({ label, value, color, bg }) => (
        <div key={label} className={`${bg} rounded-xl p-4 text-center border border-white shadow-sm`}>
          <p className={`text-2xl font-extrabold ${color}`}>{value}</p>
          <p className="text-xs text-gray-400 font-medium mt-1">{label}</p>
        </div>
      ))}
    </div>
  )
}

// ─── Loading Overlay ─────────────────────────────────────────────────────────
function LoadingSpinner({ message = 'Analyzing image…' }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4">
      <div className="relative w-16 h-16">
        <div className="absolute inset-0 rounded-full border-4 border-blue-100" />
        <div className="absolute inset-0 rounded-full border-4 border-blue-500 border-t-transparent animate-spin" />
        <div className="absolute inset-2 rounded-full border-4 border-indigo-500 border-b-transparent animate-spin" style={{ animationDirection: 'reverse', animationDuration: '0.8s' }} />
      </div>
      <div className="text-center">
        <p className="text-sm font-semibold text-gray-600">{message}</p>
        <p className="text-xs text-gray-400 mt-1">Running YOLOv8 inference…</p>
      </div>
    </div>
  )
}

// ─── Image Comparison Panel ───────────────────────────────────────────────────
function ImagePanel({ originalSrc, annotatedSrc, loadingAnnotated, filename }) {
  const [view, setView] = useState('split')

  if (!originalSrc) {
    return (
      <div className="flex flex-col items-center justify-center h-56 text-gray-300 bg-gray-50 rounded-xl border-2 border-dashed border-gray-200">
        <ScanSearch size={40} className="mb-2" />
        <p className="text-sm text-gray-400">Upload an image to see results</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      {/* View toggle */}
      <div className="flex gap-2 justify-end">
        {['split', 'original', 'annotated'].map((v) => (
          <button
            key={v}
            onClick={() => setView(v)}
            className={`text-xs px-3 py-1 rounded-full font-medium transition-colors capitalize ${
              view === v
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
            }`}
          >
            {v}
          </button>
        ))}
      </div>

      {view === 'split' && (
        <div className="grid grid-cols-2 gap-3">
          <div className="flex flex-col gap-1">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider text-center">Original</p>
            <div className="rounded-xl overflow-hidden border border-gray-100 bg-gray-50 shadow-sm">
              <img src={originalSrc} alt="Original" className="w-full object-contain max-h-72" />
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider text-center">Annotated</p>
            <div className="rounded-xl overflow-hidden border border-gray-100 bg-gray-50 shadow-sm min-h-[100px] flex items-center justify-center">
              {loadingAnnotated ? (
                <div className="flex flex-col items-center gap-2 py-8">
                  <RefreshCw size={22} className="text-blue-400 animate-spin" />
                  <p className="text-xs text-gray-400">Rendering boxes…</p>
                </div>
              ) : annotatedSrc ? (
                <img src={annotatedSrc} alt="Annotated" className="w-full object-contain max-h-72" />
              ) : (
                <div className="text-gray-300 text-center py-8">
                  <p className="text-xs">Annotated image unavailable</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {view === 'original' && (
        <div className="rounded-xl overflow-hidden border border-gray-100 shadow-sm">
          <img src={originalSrc} alt="Original" className="w-full object-contain max-h-96" />
        </div>
      )}

      {view === 'annotated' && (
        <div className="rounded-xl overflow-hidden border border-gray-100 shadow-sm min-h-[100px] flex items-center justify-center bg-gray-50">
          {loadingAnnotated ? (
            <div className="flex flex-col items-center gap-2 py-16">
              <RefreshCw size={28} className="text-blue-400 animate-spin" />
              <p className="text-sm text-gray-400">Rendering annotation…</p>
            </div>
          ) : annotatedSrc ? (
            <img src={annotatedSrc} alt="Annotated" className="w-full object-contain max-h-96" />
          ) : (
            <div className="text-gray-300 text-center py-16">
              <p className="text-sm">Annotated image unavailable</p>
            </div>
          )}
        </div>
      )}

      {filename && (
        <p className="text-center text-xs text-gray-400 truncate">{filename}</p>
      )}
    </div>
  )
}

// ─── Single Image Tab ─────────────────────────────────────────────────────────
function SingleImageTab() {
  const [file, setFile] = useState(null)
  const [originalUrl, setOriginalUrl] = useState(null)
  const [annotatedUrl, setAnnotatedUrl] = useState(null)
  const [predictions, setPredictions] = useState([])
  const [inferenceTime, setInferenceTime] = useState(undefined)
  const [imageSize, setImageSize] = useState({ w: null, h: null })
  const [loading, setLoading] = useState(false)
  const [loadingAnnotated, setLoadingAnnotated] = useState(false)
  const [error, setError] = useState(null)
  const [hasResult, setHasResult] = useState(false)

  const handleFileSelect = useCallback((selected) => {
    if (!selected) {
      setFile(null)
      setOriginalUrl(null)
      setAnnotatedUrl(null)
      setPredictions([])
      setInferenceTime(undefined)
      setImageSize({ w: null, h: null })
      setHasResult(false)
      setError(null)
      return
    }
    setFile(selected)
    setOriginalUrl(URL.createObjectURL(selected))
    setAnnotatedUrl(null)
    setPredictions([])
    setHasResult(false)
    setError(null)
  }, [])

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setAnnotatedUrl(null)
    setHasResult(false)

    try {
      const data = await detectDefects(file)
      setPredictions(data.predictions || [])
      setInferenceTime(data.inference_time_ms)
      setImageSize({ w: data.image_width, h: data.image_height })
      setHasResult(true)

      // Fetch annotated image in parallel
      setLoadingAnnotated(true)
      getAnnotatedImage(file)
        .then((url) => setAnnotatedUrl(url))
        .catch(() => setAnnotatedUrl(null))
        .finally(() => setLoadingAnnotated(false))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Upload + Analyze */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h2 className="text-base font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <span className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 text-xs font-bold flex items-center justify-center">1</span>
          Upload Image
        </h2>
        <UploadZone onFileSelect={handleFileSelect} label="Drag & drop an image here" />
        <button
          onClick={handleAnalyze}
          disabled={!file || loading}
          className="mt-4 w-full py-3 rounded-xl font-semibold text-sm transition-all
            bg-gradient-to-r from-blue-600 to-indigo-600 text-white
            hover:from-blue-700 hover:to-indigo-700
            disabled:opacity-40 disabled:cursor-not-allowed
            flex items-center justify-center gap-2 shadow-sm"
        >
          {loading ? (
            <>
              <RefreshCw size={16} className="animate-spin" />
              Analyzing…
            </>
          ) : (
            <>
              <ScanSearch size={16} />
              Detect Defects
            </>
          )}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle size={18} className="text-red-500 shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-red-700">Detection Failed</p>
            <p className="text-xs text-red-500 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <LoadingSpinner />
        </div>
      )}

      {/* Results */}
      {!loading && hasResult && (
        <>
          {/* Stats */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
            <h2 className="text-base font-semibold text-gray-800 mb-4 flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-indigo-100 text-indigo-600 text-xs font-bold flex items-center justify-center">2</span>
              Summary
            </h2>
            <StatsBar predictions={predictions} inferenceTime={inferenceTime} />
          </div>

          {/* Image comparison */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
            <h2 className="text-base font-semibold text-gray-800 mb-4 flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-purple-100 text-purple-600 text-xs font-bold flex items-center justify-center">3</span>
              Image Comparison
            </h2>
            <ImagePanel
              originalSrc={originalUrl}
              annotatedSrc={annotatedUrl}
              loadingAnnotated={loadingAnnotated}
              filename={file?.name}
            />
          </div>

          {/* Detection list */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
            <h2 className="text-base font-semibold text-gray-800 mb-4 flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-green-100 text-green-600 text-xs font-bold flex items-center justify-center">4</span>
              Detection Details
              <span className="ml-auto text-xs font-normal text-gray-400">
                Click each row to expand bbox
              </span>
            </h2>
            <DetectionResult
              predictions={predictions}
              inferenceTime={inferenceTime}
              imageWidth={imageSize.w}
              imageHeight={imageSize.h}
            />
          </div>
        </>
      )}

      {/* Empty state when image selected but not yet analyzed */}
      {!loading && !hasResult && !error && originalUrl && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <ImagePanel originalSrc={originalUrl} annotatedSrc={null} loadingAnnotated={false} filename={file?.name} />
          <p className="text-center text-sm text-gray-400 mt-4">
            Click <strong className="text-gray-600">Detect Defects</strong> to analyze this image
          </p>
        </div>
      )}
    </div>
  )
}

// ─── App Root ─────────────────────────────────────────────────────────────────
const TABS = [
  { id: 'single', label: 'Single Image', icon: ScanSearch },
  { id: 'batch', label: 'Batch Upload', icon: Layers },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('single')

  return (
    <div className="min-h-screen bg-gray-100">
      {/* ── Header ── */}
      <header className="bg-gradient-to-r from-[#0f1b35] via-[#1a2744] to-[#1e3a5f] shadow-lg sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-yellow-400 flex items-center justify-center shadow-md">
              <Zap size={20} className="text-[#0f1b35]" fill="currentColor" />
            </div>
            <div>
              <h1 className="text-lg font-extrabold text-white leading-tight tracking-tight">
                Insulator Defect Detector
              </h1>
              <p className="text-xs text-blue-300/80 font-medium">Powered by YOLOv8</p>
            </div>
          </div>
          <HealthBadge />
        </div>

        {/* Tab nav */}
        <div className="max-w-6xl mx-auto px-4 sm:px-6">
          <div className="flex gap-1 pb-0">
            {TABS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`flex items-center gap-2 px-5 py-3 text-sm font-semibold rounded-t-lg transition-all border-b-2 ${
                  activeTab === id
                    ? 'bg-white text-blue-700 border-blue-500 shadow-sm'
                    : 'text-blue-200/70 border-transparent hover:text-white hover:bg-white/10'
                }`}
              >
                <Icon size={15} />
                {label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* ── Main content ── */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
        {activeTab === 'single' && <SingleImageTab />}
        {activeTab === 'batch' && <BatchResults />}
      </main>

      {/* ── Footer ── */}
      <footer className="text-center py-6 text-xs text-gray-400">
        Insulator Defect Detector — YOLOv8 · FastAPI · React · Vite
      </footer>
    </div>
  )
}
