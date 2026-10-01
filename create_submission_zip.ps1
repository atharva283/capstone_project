<#
.SYNOPSIS
  Builds the Edureka submission ZIP:  <name>_capstone_morning_batch_13.zip

.DESCRIPTION
  * Uses a WHITELIST: only real project files go in (source code, docs, samples, tests,
    requirements.txt, .env and .env.example). Everything else is left out on purpose:
    .venv, .git, __pycache__, .kilo, .vscode, node_modules, .cache, dist, chroma_db data,
    uploaded_documents data, OS/IDE files and the course PDFs.
  * Puts everything inside ONE wrapper folder named like the ZIP (<name>_capstone_morning_batch_13/),
    so evaluators extract a single tidy folder they can open in a terminal.
  * Writes ZIP entries with forward slashes so the file opens correctly on Windows, Mac and Linux
    (Windows PowerShell's Compress-Archive does not).
  * Checks that .env holds a real Gemini key, and that no API key appears in any OTHER file.
  * Checks the ZIP is smaller than 19 MB and prints the final size.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\create_submission_zip.ps1
#>
[CmdletBinding()]
param(
    [string]$Name = 'atharvabaldota',
    [double]$MaxMegabytes = 19
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Say($tag, $text, $color) { Write-Host ("[{0}] {1}" -f $tag, $text) -ForegroundColor $color }
function Fail($text) { Say 'FAIL' $text 'Red'; exit 1 }
function IsRealKey($value) {
    $v = "$value".Trim().ToLower()
    if ($v.Length -eq 0) { return $false }
    foreach ($hint in 'your_', 'paste_', 'placeholder', 'changeme', 'xxxx') { if ($v.Contains($hint)) { return $false } }
    return $true
}

if ($Name -notmatch '^[A-Za-z0-9_]+$') { Fail "Name may contain only letters, digits and underscores (got '$Name')." }
$root = (Resolve-Path -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
$wrapper = "${Name}_capstone_morning_batch_13"
$zipName = "$wrapper.zip"
$distDir = Join-Path $root 'dist'
$zipPath = Join-Path $distDir $zipName
$limitBytes = [long]($MaxMegabytes * 1024 * 1024)

Write-Host ''
Write-Host "Building $zipName" -ForegroundColor Cyan
Write-Host '-----------------------------------------------'

# ---------------------------------------------------------------- 1. choose the files (whitelist)
$rootFiles = 'app.py', 'agent.py', 'ingestion.py', 'retriever.py', 'llm.py', 'requirements.txt',
             '.env', '.env.example', 'README.md', 'DOCUMENTATION.md', 'run_windows.bat',
             'architecture_diagram.jpg', 'README.pdf', 'DOCUMENTATION.pdf', '.streamlit/config.toml'
$junkNames = 'Thumbs.db', '.DS_Store', 'desktop.ini'
$include = New-Object 'System.Collections.Generic.List[string]'

function ToRelative($fullName) { return $fullName.Substring($root.Length).TrimStart('\', '/').Replace('\', '/') }

foreach ($file in $rootFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $file) -PathType Leaf)) { Fail "Required file is missing: $file" }
    $include.Add($file)
}
foreach ($folder in 'sample_documents', 'tests') {
    $dir = Join-Path $root $folder
    if (-not (Test-Path -LiteralPath $dir)) { Fail "Required folder is missing: $folder" }
    Get-ChildItem -LiteralPath $dir -Recurse -File -Force |
        Where-Object { $junkNames -notcontains $_.Name -and $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and $_.Extension -ne '.pyc' } |
        ForEach-Object { $include.Add((ToRelative $_.FullName)) }
}
Say 'OK' "$($include.Count) files selected (plus 2 empty folder markers)" 'Green'

# ---------------------------------------------------------------- 2. check .env and scan for key leaks
$secrets = @()
$hasGemini = $false
$hasGroq = $false
foreach ($line in Get-Content -LiteralPath (Join-Path $root '.env')) {
    if ($line -match '^\s*(GOOGLE_API_KEY|GROQ_API_KEY)\s*=\s*(.*?)\s*$') {
        $value = $Matches[2].Trim('"', "'")
        if (IsRealKey $value) {
            $secrets += $value
            if ($Matches[1] -eq 'GOOGLE_API_KEY') { $hasGemini = $true } else { $hasGroq = $true }
        }
    }
}
if (-not $hasGemini) { Fail '.env has no real GOOGLE_API_KEY. Evaluators need it to run the app.' }
Say 'OK' '.env contains a Gemini key' 'Green'
if ($hasGroq) { Say 'OK' '.env contains a Groq key (automatic fallback will work)' 'Green' }
else { Say 'WARN' '.env has NO Groq key: the fallback LLM will not work. Add GROQ_API_KEY before final submission.' 'Yellow' }

foreach ($rel in $include) {
    if ($rel -eq '.env') { continue }
    $bytes = [System.IO.File]::ReadAllBytes((Join-Path $root $rel))
    $text = [System.Text.Encoding]::GetEncoding(28591).GetString($bytes)
    foreach ($secret in $secrets) {
        if ($text.IndexOf($secret, [System.StringComparison]::Ordinal) -ge 0) { Fail "An API key was found inside '$rel'. It must only be in .env." }
    }
}
Say 'OK' 'No API key appears in any file other than .env' 'Green'

# ---------------------------------------------------------------- 3. warn about local data that stays out
foreach ($data in 'chroma_db', 'uploaded_documents') {
    $dir = Join-Path $root $data
    if (Test-Path -LiteralPath $dir) {
        $extra = @(Get-ChildItem -LiteralPath $dir -Force | Where-Object { $_.Name -ne '.gitkeep' })
        if ($extra.Count -gt 0) { Say 'INFO' "$data/ has $($extra.Count) local item(s); they are NOT included in the ZIP (only an empty .gitkeep is)." 'DarkYellow' }
    }
}

# ---------------------------------------------------------------- 4. write the ZIP
New-Item -ItemType Directory -Force -Path $distDir | Out-Null
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
$stream = [System.IO.File]::Open($zipPath, [System.IO.FileMode]::Create)
$zip = New-Object System.IO.Compression.ZipArchive($stream, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($rel in $include) {
        $entry = $zip.CreateEntry("$wrapper/$rel", [System.IO.Compression.CompressionLevel]::Optimal)
        $bytes = [System.IO.File]::ReadAllBytes((Join-Path $root $rel))
        $target = $entry.Open()
        try { $target.Write($bytes, 0, $bytes.Length) } finally { $target.Dispose() }
    }
    foreach ($marker in 'chroma_db/.gitkeep', 'uploaded_documents/.gitkeep') {
        $null = $zip.CreateEntry("$wrapper/$marker")   # empty folders that the app fills at run time
    }
} finally {
    $zip.Dispose()
    $stream.Dispose()
}

# ---------------------------------------------------------------- 5. verify the finished ZIP
$forbidden = '(^|/)(\.venv|venv|\.git|__pycache__|\.kilo|\.vscode|\.idea|node_modules|\.cache|dist)(/|$)',
             '(^|/)(Thumbs\.db|\.DS_Store|desktop\.ini)$', '\.pyc$', '\.sqlite3$', '\.bin$',
             'capstone_project\.pdf$', 'Additional instruction', '\.gitignore$', 'create_submission_zip'
$archive = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    $names = @($archive.Entries | ForEach-Object { $_.FullName })
} finally { $archive.Dispose() }
foreach ($n in $names) {
    if (-not $n.StartsWith("$wrapper/")) { Fail "Entry is outside the wrapper folder '$wrapper': $n" }
}
# From here on, look at the paths as they appear INSIDE the wrapper folder.
$names = @($names | ForEach-Object { $_.Substring($wrapper.Length + 1) })
foreach ($n in $names) {
    if ($n.Contains('\')) { Fail "Entry uses a backslash: $n" }
    foreach ($pattern in $forbidden) { if ($n -match $pattern) { Fail "Forbidden item in ZIP: $n" } }
    if ($n -match '^(chroma_db|uploaded_documents)/' -and $n -notmatch '/\.gitkeep$') { Fail "Runtime data in ZIP: $n" }
}
foreach ($must in 'app.py', 'README.md', 'DOCUMENTATION.md', 'README.pdf', 'DOCUMENTATION.pdf', 'architecture_diagram.jpg', 'requirements.txt', '.env', '.env.example',
                  'chroma_db/.gitkeep', 'uploaded_documents/.gitkeep', 'sample_documents/HR_Policy_Guide.pdf') {
    if ($names -notcontains $must) { Fail "Missing from ZIP: $must" }
}
Say 'OK' 'ZIP content verified: nothing forbidden, everything required is present' 'Green'

$size = (Get-Item -LiteralPath $zipPath).Length
if ($size -ge $limitBytes) { Fail ("ZIP is {0:N2} MB, which is NOT under the {1} MB limit." -f ($size / 1MB), $MaxMegabytes) }

Write-Host ''
Write-Host "Contents of $wrapper/ (top level):" -ForegroundColor Cyan
$names | ForEach-Object { ($_ -split '/')[0] } | Group-Object | Sort-Object Name |
    ForEach-Object { '  {0,-24} {1} file(s)' -f $_.Name, $_.Count }
Write-Host ''
Write-Host '===============================================' -ForegroundColor Green
Write-Host (" ZIP:   {0}" -f $zipPath) -ForegroundColor Green
Write-Host (" Files: {0}" -f $names.Count) -ForegroundColor Green
Write-Host (" Size:  {0:N2} MB ({1:N0} bytes) - limit is {2} MB" -f ($size / 1MB), $size, $MaxMegabytes) -ForegroundColor Green
Write-Host '===============================================' -ForegroundColor Green
Write-Host 'Submit this file to the Edureka LMS. Reminder: replace the API key(s) in .env with fresh ones first if you plan to.'
