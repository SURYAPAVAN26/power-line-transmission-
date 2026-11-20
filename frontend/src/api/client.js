import axios from 'axios'

// In production (e.g. Vercel), set VITE_API_URL to your Render backend URL:
// https://<your-service-name>.onrender.com
// In local development, falls back to '/api' handled by Vite proxy or Nginx.
const envApiUrl = import.meta.env.VITE_API_URL
const baseURL = envApiUrl ? envApiUrl.replace(/\/$/, '') : '/api'

const apiClient = axios.create({
  baseURL,
  timeout: 60000,
  headers: {
    'Accept': 'application/json',
  },
})

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred'
    return Promise.reject(new Error(message))
  }
)

export const detectDefects = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  const response = await apiClient.post('/detect', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return response.data
}

export const getAnnotatedImage = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  const response = await apiClient.post('/annotated', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob',
  })
  return URL.createObjectURL(response.data)
}

export const detectBatch = async (files) => {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))
  const response = await apiClient.post('/detect/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return response.data
}

export const getHealth = async () => {
  const response = await apiClient.get('/health')
  return response.data
}

export default apiClient
