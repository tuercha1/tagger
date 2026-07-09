$Host.UI.RawUI.WindowTitle = "Anima Tagger"

$HostName = "127.0.0.1"
$Port = 7860
$Url = "http://{0}:{1}" -f $HostName, $Port
$Root = $PSScriptRoot
$Backend = Join-Path $Root "backend"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Requirements = Join-Path $Backend "requirements.txt"

function Tag {
    param(
        [string]$Text,
        [string]$Color = "Cyan"
    )
    Write-Host $Text -ForegroundColor $Color
}

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    Tag "[$Name] ..." Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        Tag "[$Name] ERROR $LASTEXITCODE" Red
        Read-Host
        exit $LASTEXITCODE
    }
    Tag "[$Name] OK" Green
}

function Find-Python311 {
    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) {
        & py -3.11 --version *> $null
        if ($LASTEXITCODE -eq 0) {
            return ,@("py", "-3.11")
        }
    }

    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        $found = (& uv python find 3.11 2>$null | Select-Object -First 1)
        if ($found -and (Test-Path -LiteralPath $found)) {
            return ,@($found)
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        $version = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null)
        if ($version -eq "3.11") {
            return ,@("python")
        }
    }

    return $null
}

Tag "Anima Tagger" Cyan
Tag "地址  $Url" Cyan
Tag "目录  $Root" Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath (Join-Path $Backend "main.py"))) {
    Tag "[后端] ERROR" Red
    Tag "未找到 backend\main.py" Red
    Read-Host
    exit 1
}
Tag "[后端] OK" Green

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $python311 = @(Find-Python311)
    if (-not $python311) {
        Tag "[Python] ERROR" Red
        Tag "需要 Python 3.11。请安装 Python 3.11 后重试。" Red
        Read-Host
        exit 1
    }
    Tag "[虚拟环境] 创建" Cyan
    if ($python311.Count -gt 1) {
        & $python311[0] $python311[1] -m venv (Join-Path $Root ".venv")
    } else {
        & $python311[0] -m venv (Join-Path $Root ".venv")
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
        Tag "[虚拟环境] ERROR" Red
        Read-Host
        exit 1
    }
    Tag "[虚拟环境] OK" Green
}
Tag "[Python] OK" Green

Run-Step "依赖" {
    & $VenvPython -m pip install -r $Requirements
}

Tag "[端口] $Port" Cyan
try {
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction Stop }
    Tag "[端口] OK" Green
} catch {
    Tag "[端口] ERROR" Red
    Write-Error $_
}

Tag "[浏览器] 打开" Cyan
try {
    Start-Process $Url
} catch {
    Tag "[浏览器] ERROR" Red
    Write-Error $_
}

Tag "[日志] 警告/错误" Cyan
Tag "[服务] 运行中" Magenta
Write-Host ""

& $VenvPython -m uvicorn main:app --app-dir $Backend --host $HostName --port $Port --log-level warning --no-access-log
$ExitCode = $LASTEXITCODE

Write-Host ""
if ($ExitCode -ne 0) {
    Tag "[服务] ERROR $ExitCode" Red
} else {
    Tag "[停止] OK" Green
}

Read-Host
exit $ExitCode
