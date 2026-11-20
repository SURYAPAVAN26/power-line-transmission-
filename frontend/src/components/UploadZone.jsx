import { useCallback, useRef, useState } from 'react'
import { Upload, ImagePlus, X } from 'lucide-react'

export default function UploadZone({ onFileSelect, label = 'Upload Image', accept = '.jpg,.jpeg,.png', multiple = false }) {
  const [isDragging, setIsDragging] = useState(false)
  const [preview, setPreview] = useState(null)
  const [fileName, setFileName] = useState(null)
  const inputRef = useRef(null)

  const handleFiles = useCallback(
    (files) => {
      if (!files || files.length === 0) return
      if (multiple) {
        const validFiles = Array.from(files).filter((f) =>
          ['image/jpeg', 'image/png', 'image/jpg'].includes(f.type)
        )
        if (validFiles.length === 0) return
        onFileSelect(validFiles)
        setFileName(`${validFiles.length} image${validFiles.length > 1 ? 's' : ''} selected`)
        setPreview(null)
      } else {
        const file = files[0]
        if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) return
        setFileName(file.name)
        const reader = new FileReader()
        reader.onload = (e) => setPreview(e.target.result)
        reader.readAsDataURL(file)
        onFileSelect(file)
      }
    },
    [onFileSelect, multiple]
  )

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      setIsDragging(false)
      handleFiles(e.dataTransfer.files)
    },
    [handleFiles]
  )

  const onDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }
  const onDragLeave = () => setIsDragging(false)

  const onInputChange = (e) => handleFiles(e.target.files)

  const clearSelection = (e) => {
    e.stopPropagation()
    setPreview(null)
    setFileName(null)
    if (inputRef.current) inputRef.current.value = ''
    onFileSelect(null)
  }

  return (
    <div
      className={`relative rounded-xl border-2 border-dashed transition-all duration-200 cursor-pointer
        ${isDragging
          ? 'border-blue-400 bg-blue-50 scale-[1.01]'
          : 'border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-blue-50/50'
        }
        ${preview ? 'p-2' : 'p-8'}
      `}
      onClick={() => inputRef.current?.click()}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept={accept}
        multiple={multiple}
        onChange={onInputChange}
      />

      {preview ? (
        <div className="relative group">
          <img
            src={preview}
            alt="Preview"
            className="w-full max-h-56 object-contain rounded-lg"
          />
          <button
            onClick={clearSelection}
            className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity shadow-md"
          >
            <X size={14} />
          </button>
          <p className="text-center text-xs text-gray-500 mt-1 truncate px-2">{fileName}</p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 select-none">
          <div className={`p-4 rounded-full transition-colors ${isDragging ? 'bg-blue-100' : 'bg-white shadow-sm'}`}>
            {isDragging ? (
              <Upload size={32} className="text-blue-500" />
            ) : (
              <ImagePlus size={32} className="text-gray-400" />
            )}
          </div>
          <div className="text-center">
            <p className="text-sm font-semibold text-gray-700">
              {isDragging ? 'Drop it here!' : label}
            </p>
            <p className="text-xs text-gray-400 mt-1">
              {multiple ? 'JPG, PNG — select multiple files' : 'JPG or PNG — drag & drop or click'}
            </p>
          </div>
          {fileName && (
            <div className="flex items-center gap-1 bg-blue-100 text-blue-700 text-xs rounded-full px-3 py-1">
              <span className="truncate max-w-[200px]">{fileName}</span>
              <button onClick={clearSelection} className="ml-1 hover:text-red-500">
                <X size={12} />
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
