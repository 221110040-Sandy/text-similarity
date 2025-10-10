#!/bin/bash
echo "🚀 Starting Text Similarity Application"
echo "1. Starting FastAPI backend..."
python model_api.py &

echo "2. Waiting for backend to initialize..."
sleep 10

echo "3. Starting Streamlit frontend..."
streamlit run app_frontend.py

echo "✅ Applications started!"
echo "- Backend API: http://localhost:8000"
echo "- Frontend UI: http://localhost:8501"