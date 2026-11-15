$TaskId = $null
$OutLog = $null
$argv = @()
$i = 0
while ($i -lt $args.Count) {
    $token = [string]$args[$i]
    if ($token -eq '-TaskId' -and ($i + 1) -lt $args.Count) {
        $TaskId = [string]$args[$i + 1]
        $i += 2
    } elseif ($token -eq '-OutLog' -and ($i + 1) -lt $args.Count) {
        $OutLog = [string]$args[$i + 1]
        $i += 2
    } elseif ($token -eq '--') {
        $i += 1
    } else {
        $argv += $token
        $i += 1
    }
}
if ([string]::IsNullOrEmpty($TaskId) -or [string]::IsNullOrEmpty($OutLog)) {
    Write-Error "usage: nice_run.ps1 -TaskId <id> -OutLog <path> -- <command...>"
    exit 2
}
if ($argv.Count -eq 0) {
    "STOP task=$TaskId empty command" | Tee-Object -FilePath $OutLog
    exit 2
}
$loadValue = -1
$sample = Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty CounterSamples | Select-Object -First 1
if ($null -ne $sample) {
    $loadValue = [math]::Round($sample.CookedValue, 1)
}
if ($loadValue -ge 0 -and $loadValue -gt 50) {
    "STOP task=$TaskId load=$loadValue" | Tee-Object -FilePath $OutLog
    exit 2
}
$parts = @()
for ($i = 1; $i -lt $argv.Count; $i++) {
    $text = [string]$argv[$i]
    if ($text -eq "" -or $text.Contains(" ") -or $text.Contains("`t") -or $text.Contains('"')) {
        $text = '"' + $text.Replace('"', '\"') + '"'
    }
    $parts += $text
}
if ($loadValue -ge 0) {
    $header = "START task=$TaskId load=$loadValue cmd=$($argv[0]) $($parts -join ' ')"
} else {
    $header = "START task=$TaskId load=unknown cmd=$($argv[0]) $($parts -join ' ')"
}
$header | Tee-Object -FilePath $OutLog
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $argv[0]
$psi.Arguments = ($parts -join " ")
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $false
try {
    $proc = [System.Diagnostics.Process]::Start($psi)
} catch {
    "STOP task=$TaskId launch failed" | Tee-Object -FilePath $OutLog -Append
    exit 2
}
try {
    $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
} catch {
}
$proc.WaitForExit()
$code = $proc.ExitCode
"END task=$TaskId exit=$code" | Tee-Object -FilePath $OutLog -Append
exit $code
