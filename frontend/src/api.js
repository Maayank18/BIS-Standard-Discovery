import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({ baseURL: API_BASE, timeout: 30000 });

export const queryStandards = (query, topK = 5, generateRationales = true) =>
  api.post('/query', { query, top_k: topK, generate_rationales: generateRationales })
     .then(r => r.data);

export const getHealth = () => api.get('/health').then(r => r.data);

export const getCategories = () => api.get('/categories').then(r => r.data);

export const getStandards = (params = {}) =>
  api.get('/standards', { params }).then(r => r.data);

export const getExamples = () => api.get('/examples').then(r => r.data);
