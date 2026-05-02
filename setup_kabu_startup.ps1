# setup_kabu_startup.ps1
# kabuステーション(R) PC起動時自動起動設定スクリプト
#
# 実行タイミング:
#   口座開設 + kabuステーション(R)インストール完了後に1回だけ実行する
#   (管理者権限不要)
#
# 実行方法:
#   PowerShell で:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#   .\setup_kabu_startup.ps1
#
# 設定内容:
#   Windowsタスクスケジューラに「kabuStation_AutoStart」タスクを登録
#   ログオン時に自動起動 → 17:00の自動売買スクリプト実行前に確実に起動

param(
    [string]$KabuStationExe = "",  # 手動指定（省略時は自動検索）
    [switch]$Uninstall             # タスク削除
)

$TaskName = "kabuStation_AutoStart"

# ── タスク削除モード ────────────────────────────────────────
if ($Uninstall) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "[OK] タスク「$TaskName」を削除しました。" -ForegroundColor Green
    } else {
        Write-Host "[INFO] タスク「$TaskName」は登録されていません。" -ForegroundColor Yellow
    }
    exit 0
}

# ── kabuStation.exe の検索 ──────────────────────────────────
$SearchPaths = @(
    $KabuStationExe,
    "$env:LOCALAPPDATA\kabustation\kabuStation.exe",
    "$env:LOCALAPPDATA\kabu_station\kabuStation.exe",
    "C:\Program Files\kabustation\kabuStation.exe",
    "C:\Program Files (x86)\kabustation\kabuStation.exe",
    "$env:LOCALAPPDATA\Programs\kabustation\kabuStation.exe"
)

$ExePath = $null
foreach ($path in $SearchPaths) {
    if ($path -and (Test-Path $path)) {
        $ExePath = $path
        break
    }
}

if (-not $ExePath) {
    # スタートメニューから検索
    $StartMenu = @(
        "$env:APPDATA\Microsoft\Windows\Start Menu\Programs",
        "C:\ProgramData\Microsoft\Windows\Start Menu\Programs"
    )
    foreach ($dir in $StartMenu) {
        $lnk = Get-ChildItem $dir -Recurse -Filter "*kabu*Station*.lnk" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($lnk) {
            $shell = New-Object -ComObject WScript.Shell
            $target = $shell.CreateShortcut($lnk.FullName).TargetPath
            if (Test-Path $target) {
                $ExePath = $target
                break
            }
        }
    }
}

if (-not $ExePath) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host " kabuStation.exe が見つかりません" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "以下を確認してください:"
    Write-Host "  1. kabuステーション(R) をインストールしたか"
    Write-Host "     https://kabu.com/kabustation/"
    Write-Host ""
    Write-Host "  2. インストール済みの場合、パスを手動指定:"
    Write-Host "     .\setup_kabu_startup.ps1 -KabuStationExe 'C:\path\to\kabuStation.exe'"
    exit 1
}

Write-Host ""
Write-Host "  kabuStation.exe: $ExePath" -ForegroundColor Cyan
Write-Host ""

# ── 既存タスク確認 ──────────────────────────────────────────
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[INFO] タスク「$TaskName」は既に登録済みです。" -ForegroundColor Yellow
    Write-Host "  削除して再登録する場合: .\setup_kabu_startup.ps1 -Uninstall"
    Write-Host "  その後このスクリプトを再実行してください。"
    exit 0
}

# ── タスクスケジューラに登録 ────────────────────────────────
$action  = New-ScheduledTaskAction -Execute $ExePath
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit "00:00:00" `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable

try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description "kabuステーション(R) ログオン時自動起動（自動売買システム用）" `
        -RunLevel Limited `
        -Force | Out-Null

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  設定完了" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  タスク名 : $TaskName"
    Write-Host "  起動タイミング: ログオン時"
    Write-Host "  起動対象 : $ExePath"
    Write-Host ""
    Write-Host "次回PC再起動後、kabuステーション(R)が自動起動します。"
    Write-Host ""
    Write-Host "[重要] kabuステーション(R)の自動ログイン設定も忘れずに:"
    Write-Host "  kabuステーション(R) → システム設定 → 起動設定 → 自動ログインにチェック"

} catch {
    Write-Host ""
    Write-Host "[ERROR] タスク登録に失敗しました: $_" -ForegroundColor Red
    Write-Host "管理者権限で実行する必要がある場合は、PowerShellを管理者として起動してください。"
    exit 1
}
