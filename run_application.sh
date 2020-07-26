#! /usr/bin/env bash

# exit in case of some error
set -e
# start the backend and then the frontend application
uvicorn app.main:app --reload --port 8080 & pid1="$!"; streamlit run frontend/app.py --server.port 8084; kill $pid1
