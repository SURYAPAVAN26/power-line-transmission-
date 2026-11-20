import { ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'

const CLASS_STYLES = {
  defect: {
    badge: 'bg-red-100 text-red-700 border border-red-200',
    bar: 'bg-red-500',
    dot: 'bg-red-500',
  },
  insulator: {
    badge: 'bg-green-100 text-green-700 border border-green-200',
    bar: 'bg-green-500',
    dot: 'bg-green-500',
  },
}

function getStyle(className) {
  return CLASS_STYLES[className?.toLowerCase()] || {
    badge: 'bg-gray-100 text-gray-700 border border-gray-200',
    bar: 'bg-gray-500',
    dot: 'bg-gray-500',
  }
}

function DetectionItem({ prediction, index }) {
  const [expanded, setExpanded] = useState(false)
  const style = getStyle(prediction.class_name)
  const confidencePct = Math.round((prediction.confidence || 0) * 100)
  const [x1, y1, x2, y2] = prediction.bbox || [0, 0, 0, 0]
  const w = Math.round(x2 - x1)
  const h = Math.round(y2 - y1)

  return (
    <div className="rounded-lg border border-gray-100 bg-white shadow-sm overflow-hidden">
      <div
        className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-gray-50 transition-colors"
        onClick={() => setExpanded((v) => !v)}
      >
        <span className="text-xs font-bold text-gray-400 w-5 shrink-0">#{index + 1}</span>
        <div className={`w-2.5 h-2.5 rounded-full shrink-0 ${style.dot}`} />
        <span className={`text-xs font-semibold uppercase tracking-wide rounded-full px-2.5 py-0.5 ${style.badge}`}>
          {prediction.class_name}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <div className="flex-1 bg-gray-100 rounded-full h-1.5 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${style.bar}`}
                style={{ width: `${confidencePct}%` }}
              />
            </div>
            <span className="text-xs font-bold text-gray-600 shrink-0 w-9 text-right">
              {confidencePct}%
            </span>
          </div>
        </div>
        <div className="text-gray-400 shrink-0">
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </div>
      </div>

      {expanded && (
        <div className="px-4 py-3 bg-gray-50 border-t border-gray-100">
          <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider">Bounding Box</p>
          <div className="grid grid-cols-2 gap-2">
            {[
              { label: 'x1', value: Math.round(x1) },
              { label: 'y1', value: Math.round(y1) },
              { label: 'x2', value: Math.round(x2) },
              { label: 'y2', value: Math.round(y2) },
            ].map(({ label, value }) => (
              <div key={label} className="flex items-center justify-between bg-white rounded-md px-3 py-1.5 border border-gray-100">
                <span className="text-xs text-gray-400 font-medium">{label}</span>
                <span className="text-xs font-mono font-bold text-gray-700">{value}px</span>
              </div>
            ))}
          </div>
          <div className="flex gap-2 mt-2">
            <div className="flex items-center justify-between flex-1 bg-white rounded-md px-3 py-1.5 border border-gray-100">
              <span className="text-xs text-gray-400 font-medium">W</span>
              <span className="text-xs font-mono font-bold text-gray-700">{w}px</span>
            </div>
            <div className="flex items-center justify-between flex-1 bg-white rounded-md px-3 py-1.5 border border-gray-100">
              <span className="text-xs text-gray-400 font-medium">H</span>
              <span className="text-xs font-mono font-bold text-gray-700">{h}px</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default function DetectionResult({ predictions = [], inferenceTime, imageWidth, imageHeight }) {
  if (!predictions || predictions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-400">
        <div className="w-14 h-14 rounded-full bg-gray-100 flex items-center justify-center mb-3">
          <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <p className="text-sm font-medium">No detections found</p>
        <p className="text-xs text-gray-300 mt-1">The model found no objects in this image</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      {(inferenceTime !== undefined || imageWidth) && (
        <div className="flex flex-wrap gap-2 mb-1">
          {inferenceTime !== undefined && (
            <span className="text-xs bg-blue-50 text-blue-600 border border-blue-100 rounded-full px-2.5 py-0.5 font-medium">
              ⏱ {typeof inferenceTime === 'number' ? inferenceTime.toFixed(1) : inferenceTime} ms
            </span>
          )}
          {imageWidth && imageHeight && (
            <span className="text-xs bg-purple-50 text-purple-600 border border-purple-100 rounded-full px-2.5 py-0.5 font-medium">
              🖼 {imageWidth}×{imageHeight}
            </span>
          )}
        </div>
      )}
      {predictions.map((pred, i) => (
        <DetectionItem key={i} prediction={pred} index={i} />
      ))}
    </div>
  )
}
