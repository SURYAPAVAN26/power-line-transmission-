import { useState } from 'react'
import { detectBatch, getAnnotatedImage } from '../api/client'
import UploadZone from './UploadZone'
import DetectionResult from './DetectionResult'
import { RefreshCw, AlertCircle, Image, ChevronDown, ChevronUp } from 'lucide-react'

function BatchCard({ result, index }) {
  const [expanded, setExpanded] = useState(false)
  const [annotatedUrl, setAnnotatedUrl] = useState(null)
  const [loadingAnnotated, setLoadingAnnotated] = useState(false)
  const predictions = result.predictions || []
  const defects = predictions.filter((p) => p.class_name?.toLowerCase() === 'defect')
  const insulators = predictions.filter((p) => p.class_name?.toLowerCase() === 'insulator')

  const loadAnnotated = async () => {
    if (annotatedUrl || !result._file) return
    setLoadingAnnotated(true)
    try {
      const url = await getAnnotatedImage(result._file)
      setAnnotatedUrl(url)
    } catch {
      // silent
    } finally {
      setLoadingAnnotated(false)
    }
  }

  return (
    <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-100 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Image size={15} className="text-gray-400 shrink-0" />
          <span className="text-sm font-semibold text-gray-700 truncate">{result.filename}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {defects.length > 0 && (
            <span className="text-xs bg-red-100 text-red-700 border border-red-200 rounded-full px-2 py-0.5 font-semibold">
              {defects.length} defect{defects.length > 1 ? 's' : ''}
            </span>
          )}
          {insulators.length > 0 && (
            <span className="text-xs bg-green-100 text-green-700 border border-green-200 rounded-full px-2 py-0.5 font-semibold">
              {insulators.length} insulator{insulators.length > 1 ? 's' : ''}
            </span>
          )}
          {predictions.length === 0 && (
            <span className="text-xs bg-gray-100 text-gray-500 rounded-full px-2 py-0.5">
              No detections
            </span>
          )}
        </div>
      </div>

      <div className="p-4">
        {/* Stats row */}
        <div className="flex gap-3 mb-3">
          <div className="flex-1 bg-gray-50 rounded-lg p-2 text-center">
            <p className="text-lg font-bold text-gray-800">{predictions.length}</p>
            <p className="text-xs text-gray-400">Total</p>
          </div>
          <div className="flex-1 bg-red-50 rounded-lg p-2 text-center">
            <p className="text-lg font-bold text-red-600">{defects.length}</p>
            <p className="text-xs text-red-400">Defects</p>
          </div>
          <div className="flex-1 bg-green-50 rounded-lg p-2 text-center">
            <p className="text-lg font-bold text-green-600">{insulators.length}</p>
            <p className="text-xs text-green-400">Insulators</p>
          </div>
          {result.inference_time_ms !== undefined && (
            <div className="flex-1 bg-blue-50 rounded-lg p-2 text-center">
              <p className="text-lg font-bold text-blue-600">{result.inference_time_ms.toFixed(0)}</p>
              <p className="text-xs text-blue-400">ms</p>
            </div>
          )}
        </div>

        {/* Annotated image toggle */}
        {result._file && (
          <button
            onClick={() => { loadAnnotated(); setExpanded((v) => !v) }}
            className="w-full flex items-center justify-center gap-2 text-xs text-blue-600 hover:text-blue-700 py-2 border border-blue-100 rounded-lg hover:bg-blue-50 transition-colors mb-3"
          >
            {loadingAnnotated ? (
              <RefreshCw size={12} className="animate-spin" />
            ) : expanded ? (
              <ChevronUp size={12} />
            ) : (
              <ChevronDown size={12} />
            )}
            {expanded ? 'Hide annotated' : 'Show annotated image'}
          </button>
        )}

        {expanded && annotatedUrl && (
          <div className="mb-3">
            <img src={annotatedUrl} alt="Annotated" className="w-full rounded-lg object-contain max-h-48 border border-gray-100" />
          </div>
        )}

        {/* Predictions */}
        {predictions.length > 0 && (
          <DetectionResult predictions={predictions} />
        )}
      </div>
    </div>
  )
}

export default function BatchResults() {
  const [files, setFiles] = useState(null)
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileSelect = (selected) => {
    setFiles(selected)
    setResults([])
    setError(null)
  }

  const handleAnalyze = async () => {
    if (!files || files.length === 0) return
    setLoading(true)
    setError(null)
    try {
      const data = await detectBatch(files)
      // Attach original file to each result for annotated image fetching
      const enriched = (data || []).map((r, i) => ({
        ...r,
        _file: files[i] || null,
      }))
      setResults(enriched)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const totalDetections = results.reduce((s, r) => s + (r.predictions?.length || 0), 0)
  const totalDefects = results.reduce(
    (s, r) => s + (r.predictions?.filter((p) => p.class_name?.toLowerCase() === 'defect').length || 0),
    0
  )
  const totalInsulators = results.reduce(
    (s, r) => s + (r.predictions?.filter((p) => p.class_name?.toLowerCase() === 'insulator').length || 0),
    0
  )

  return (
    <div className="flex flex-col gap-6">
      {/* Upload area */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h2 className="text-base font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <span className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 text-xs font-bold flex items-center justify-center">1</span>
          Select Multiple Images
        </h2>
        <UploadZone
          onFileSelect={handleFileSelect}
          label="Drag & drop multiple images"
          multiple
        />
        <button
          onClick={handleAnalyze}
          disabled={!files || loading}
          className="mt-4 w-full py-3 rounded-xl font-semibold text-sm transition-all
            bg-gradient-to-r from-blue-600 to-indigo-600 text-white
            hover:from-blue-700 hover:to-indigo-700
            disabled:opacity-40 disabled:cursor-not-allowed
            flex items-center justify-center gap-2 shadow-sm"
        >
          {loading ? (
            <>
              <RefreshCw size={16} className="animate-spin" />
              Analyzing batch…
            </>
          ) : (
            `Analyze ${files ? `${files.length} Image${files.length > 1 ? 's' : ''}` : 'Images'}`
          )}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle size={18} className="text-red-500 shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-red-700">Batch Analysis Failed</p>
            <p className="text-xs text-red-500 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Summary stats */}
      {results.length > 0 && (
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: 'Images', value: results.length, color: 'purple' },
            { label: 'Total', value: totalDetections, color: 'gray' },
            { label: 'Defects', value: totalDefects, color: 'red' },
            { label: 'Insulators', value: totalInsulators, color: 'green' },
          ].map(({ label, value, color }) => (
            <div key={label} className={`bg-${color}-50 rounded-xl p-3 text-center border border-${color}-100`}>
              <p className={`text-2xl font-bold text-${color}-700`}>{value}</p>
              <p className={`text-xs text-${color}-400 font-medium mt-0.5`}>{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Result cards */}
      {results.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {results.map((result, i) => (
            <BatchCard key={i} result={result} index={i} />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && results.length === 0 && !error && (
        <div className="text-center py-16 text-gray-300">
          <div className="text-5xl mb-3">🖼</div>
          <p className="text-sm font-medium text-gray-400">Upload images above to analyze a batch</p>
        </div>
      )}
    </div>
  )
}
