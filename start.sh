#!/bin/sh

# Jalankan aplikasi FastAPI menggunakan uvicorn
# Jika $PORT tidak tersedia, fallback ke 8000
uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}
