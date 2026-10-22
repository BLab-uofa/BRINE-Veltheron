@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Create/activate .venv and install requirements first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run dashboard.py -- %*
