@echo off
echo [!] Starting the server...
echo [?] Ctrl + C to stop the server.
uvicorn app:app --port 81 --reload
