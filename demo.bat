@echo off
REM ====================================================================
REM  DOUBLE-CLICK THIS FILE to run the whole project and open every UI.
REM
REM  Starts Docker if needed, brings up Postgres, Airflow, MLflow, the
REM  prediction API and the dashboard, loads the warehouse if it is empty,
REM  then opens all four UIs as browser tabs.
REM
REM  All the real work is in demo.ps1 next to this file.
REM ====================================================================
title E-Commerce Churn MLOps - Demo Launcher
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo.ps1"
if errorlevel 1 (
  echo.
  echo The launcher exited with an error. Scroll up for the reason.
  pause
)
