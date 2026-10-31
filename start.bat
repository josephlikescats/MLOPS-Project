@echo off
REM Double-click this file, or drag it into a terminal, to bring the whole
REM project up. All the real work is in start.ps1 next to this file.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 pause
