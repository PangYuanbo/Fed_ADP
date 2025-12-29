@echo off
REM GIA Auto Attack - Quick Launch Script
REM 一键启动 GIA 自动攻击

echo ================================================================================
echo GIA Auto Attack Tool - Quick Launch
echo 梯度反演攻击自动化工具 - 快速启动
echo ================================================================================
echo.

cd /d %~dp0\..

echo [Info] Current directory: %CD%
echo.

REM 检查 Python 环境
python --version >nul 2>&1
if errorlevel 1 (
    echo [Error] Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo [Check] Python found
echo.

REM 检查 pretrain 目录
if not exist "pretrain" (
    echo [Warning] Pretrain directory not found!
    echo [Info] Please train some models first using:
    echo        python main.py -algo FedCP -data cifar-10-normal -nc 10 -gr 200
    echo.
    pause
    exit /b 1
)

echo [Check] Pretrain directory exists
echo.

REM 显示选项
echo [Menu] Please select attack mode:
echo   1. Quick Attack (5 samples, 5000 iterations) - Recommended
echo   2. Full Attack  (5 samples, 10000 iterations) - High quality
echo   3. Custom Attack (specify parameters)
echo   4. Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto quick
if "%choice%"=="2" goto full
if "%choice%"=="3" goto custom
if "%choice%"=="4" goto end

echo [Error] Invalid choice!
pause
exit /b 1

:quick
echo.
echo [Mode] Quick Attack Mode
echo [Config] 5 samples per model, 5000 iterations
echo.
python Gradient_Inversion_Attack/quick_gia_attack.py
goto end

:full
echo.
echo [Mode] Full Attack Mode
echo [Config] 5 samples per model, 10000 iterations
echo.
python Gradient_Inversion_Attack/gia_auto_attack.py --scan_all --num_samples 5 --iterations 10000 --save_visuals
goto end

:custom
echo.
echo [Mode] Custom Attack Mode
echo.
set /p samples="Number of samples per model (default=5): "
set /p iterations="Number of iterations (default=10000): "
set /p pretrain_dir="Pretrain directory (Enter for all, or specify path): "

if "%samples%"=="" set samples=5
if "%iterations%"=="" set iterations=10000

if "%pretrain_dir%"=="" (
    echo [Custom] Attacking all models with %samples% samples, %iterations% iterations
    python Gradient_Inversion_Attack/gia_auto_attack.py --scan_all --num_samples %samples% --iterations %iterations% --save_visuals
) else (
    echo [Custom] Attacking models in %pretrain_dir% with %samples% samples, %iterations% iterations
    python Gradient_Inversion_Attack/gia_auto_attack.py --pretrain_dir %pretrain_dir% --num_samples %samples% --iterations %iterations% --save_visuals
)
goto end

:end
echo.
echo ================================================================================
echo [Complete] Attack finished!
echo [Results] Check gia_auto_results/ directory for results and visualizations
echo ================================================================================
pause
