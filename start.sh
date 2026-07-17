#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "🔑 Loading environment variables from .env file..."
    # Read .env file line by line, ignore comments and empty lines, and export
    while IFS= read -r line || [ -n "$line" ]; do
        # Ignore comments and empty lines
        if [[ ! "$line" =~ ^# ]] && [[ ! "$line" =~ ^$ ]]; then
            # Clean outer quotes if any
            clean_line=$(echo "$line" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
            export "$clean_line"
        fi
    done < .env
fi

# Terminate background processes on exit
cleanup() {
    echo ""
    echo "Terminating AETHER servers..."
    kill "$BACKEND_PID" 2>/dev/null
    kill "$FRONTEND_PID" 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "============================================="
echo "      AETHER // AI DUBBING QC DASHBOARD      "
echo "============================================="

# 1. Start backend
echo "🚀 Starting FastAPI Backend..."
cd backend
./venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# 2. Start frontend
echo "🚀 Starting Vite Frontend..."
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173 > /dev/null 2>&1 &
FRONTEND_PID=$!
cd ..

# Sleep for a moment to let servers initialize
sleep 2

echo "---------------------------------------------"
echo "✨ AETHER QC Dashboard is ready!"
echo "👉 Dashboard URL: http://localhost:5173"
echo "👉 Backend API:   http://localhost:8000"
echo "👉 Context Evaluation Report: file:///Users/choisoyeong/.gemini/antigravity-cli/brain/787cc277-908f-4c9f-adbc-779d90b476c2/qc_system_evaluation.md"
echo ""
echo "Press Ctrl+C to stop all servers."
echo "============================================="

# Keep script running
wait
