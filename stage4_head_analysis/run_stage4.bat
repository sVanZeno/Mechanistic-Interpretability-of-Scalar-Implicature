@echo off
echo ============================================================
echo  Stage 4 -- L01H03 Deep Analysis
echo ============================================================
cd /d C:\Users\zenop\Desktop\MLLM
call .venv-mi\Scripts\activate.bat
echo Venv activated. Starting experiment...
echo.
python stage4_head_analysis\analyze_L01H03.py > stage4_head_analysis\run_log.txt 2>&1
echo.
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS. Results in stage4_head_analysis\
) else (
    echo ERROR occurred. Check stage4_head_analysis\run_log.txt for details.
)
echo ============================================================
pause
