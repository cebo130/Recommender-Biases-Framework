# Build thesis.pdf — requires MiKTeX or TeX Live (pdflatex + bibtex on PATH or below).
# Usage: .\build_thesis.ps1
# After installing MiKTeX, close and reopen PowerShell (or log out) so PATH updates.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$candidates = @(
    "pdflatex",
    "${env:ProgramFiles}\MiKTeX\miktex\bin\x64\pdflatex.exe",
    "${env:ProgramFiles(x86)}\MiKTeX\miktex\bin\pdflatex.exe",
    "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
    "C:\texlive\2024\bin\windows\pdflatex.exe",
    "C:\texlive\2023\bin\windows\pdflatex.exe"
)

$pdflatex = $null
foreach ($c in $candidates) {
    if ($c -eq "pdflatex") {
        $cmd = Get-Command pdflatex -ErrorAction SilentlyContinue
        if ($cmd) { $pdflatex = $cmd.Source; break }
    } elseif (Test-Path -LiteralPath $c) {
        $pdflatex = $c
        break
    }
}

if (-not $pdflatex) {
    Write-Host @"
pdflatex was not found.

Install a LaTeX distribution, then reopen this terminal:

  MiKTeX (Windows): https://miktex.org/download
    - Choose 'Install missing packages on-the-fly: Yes'
    - After install, ensure 'Add to PATH' was selected (or add ...\miktex\bin\x64 yourself)

  TeX Live: https://tug.org/texlive/windows.html

Or use Overleaf: upload thesis.tex + references.bib and set the main file to thesis.tex.

"@
    exit 1
}

$bin = Split-Path $pdflatex -Parent
$bibtex = Join-Path $bin "bibtex.exe"
if (-not (Test-Path $bibtex)) { $bibtex = "bibtex" }

Write-Host "Using: $pdflatex"
& $pdflatex -interaction=nonstopmode thesis.tex
if (-not (Test-Path "thesis.aux")) { throw "thesis.aux missing — pdflatex failed" }
& $bibtex thesis
& $pdflatex -interaction=nonstopmode thesis.tex
& $pdflatex -interaction=nonstopmode thesis.tex

if (Test-Path "thesis.pdf") {
    Write-Host "OK: thesis.pdf created."
    Get-Item thesis.pdf | Format-List FullName, Length, LastWriteTime
} else {
    Write-Warning "thesis.pdf not found — check thesis.log for errors"
    exit 1
}
