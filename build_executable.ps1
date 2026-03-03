# Build Script for AlgoSwitcher Executable
# This script builds a standalone executable using PyInstaller

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AlgoSwitcher Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check if Python is available
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow

# Try to find Python
$pythonCmd = $null
$pythonPaths = @(
    "python",
    "py",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "C:\Python312\python.exe",
    "C:\Python311\python.exe",
    "C:\Python310\python.exe"
)

foreach ($path in $pythonPaths) {
    try {
        $testOutput = & $path --version 2>&1
        if ($LASTEXITCODE -eq 0 -or $testOutput -match "Python") {
            $pythonCmd = $path
            Write-Host "  OK Found: $testOutput" -ForegroundColor Green
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Host "  X Error: Python not found" -ForegroundColor Red
    Write-Host "  Please install Python from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# Check/Install required packages
Write-Host ""
Write-Host "[2/6] Checking required packages..." -ForegroundColor Yellow

$requiredPackages = @("pyinstaller", "pandas", "openpyxl")
foreach ($package in $requiredPackages) {
    $installed = & $pythonCmd -m pip show $package 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Installing $package..." -ForegroundColor Cyan
        & $pythonCmd -m pip install $package
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  X Failed to install $package" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "  OK $package is installed" -ForegroundColor Green
    }
}

# Clean previous build
Write-Host ""
Write-Host "[3/6] Cleaning previous build..." -ForegroundColor Yellow
if (Test-Path "build\AlgoSwitcher") {
    Remove-Item "build\AlgoSwitcher" -Recurse -Force
    Write-Host "  OK Removed old build directory" -ForegroundColor Green
}
if (Test-Path "dist\AlgoSwitcher.exe") {
    Remove-Item "dist\AlgoSwitcher.exe" -Force
    Write-Host "  OK Removed old executable" -ForegroundColor Green
}

# Update spec file with data files
Write-Host ""
Write-Host "[4/6] Preparing spec file..." -ForegroundColor Yellow

$specContent = @"
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['switch_algo.py'],
    pathex=[],
    binaries=[],
    datas=[('abtesting_instructions', 'abtesting_instructions')],
    hiddenimports=['pandas', 'openpyxl'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AlgoSwitcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
"@

Set-Content -Path "AlgoSwitcher.spec" -Value $specContent
Write-Host "  OK Spec file updated" -ForegroundColor Green

# Build executable
Write-Host ""
Write-Host "[5/6] Building executable with PyInstaller..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Cyan
Write-Host ""

& $pythonCmd -m PyInstaller AlgoSwitcher.spec --clean --noconfirm

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  X Build failed!" -ForegroundColor Red
    exit 1
}

# Verify output
Write-Host ""
Write-Host "[6/6] Verifying build..." -ForegroundColor Yellow

if (Test-Path "dist\AlgoSwitcher.exe") {
    $fileSize = (Get-Item "dist\AlgoSwitcher.exe").Length / 1MB
    Write-Host "  OK Executable created successfully!" -ForegroundColor Green
    Write-Host "  Location: dist\AlgoSwitcher.exe" -ForegroundColor Cyan
    Write-Host "  Size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Cyan
} else {
    Write-Host "  X Executable not found!" -ForegroundColor Red
    exit 1
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your executable is ready at:" -ForegroundColor White
Write-Host "  $ScriptDir\dist\AlgoSwitcher.exe" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now:" -ForegroundColor White
Write-Host "  1. Test the executable by running it" -ForegroundColor Gray
Write-Host "  2. Distribute the standalone .exe file" -ForegroundColor Gray
Write-Host "  3. Note: Run as Administrator for full functionality" -ForegroundColor Yellow
Write-Host ""
