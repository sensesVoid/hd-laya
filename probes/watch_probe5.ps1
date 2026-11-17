$out = "C:\Users\geral\hd-laya\results\kaggle_probe5"
$deadline = (Get-Date).AddMinutes(40)
$slug = "sensesvoid/hd-laya-probe5"
$fails = 0
$s = ""
while ((Get-Date) -lt $deadline) {
    try { $s = (kaggle kernels status $slug 2>&1 | Out-String) } catch { $s = "" }
    if ($s -cmatch 'KernelWorkerStatus\.([A-Z]+)') {
        $st = $Matches[1]
        $fails = 0
        Write-Output ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $st)
        if ($st -eq 'COMPLETE' -or $st -eq 'ERROR' -or $st -eq 'CANCELLED') { break }
    } else {
        $fails++
        Write-Output ("[{0}] poll-fail #{1}" -f (Get-Date -Format HH:mm:ss), $fails)
        if ($fails -ge 5) { Write-Output "giving up after 5 consecutive poll failures"; break }
        Start-Sleep -Seconds 60
        continue
    }
    Start-Sleep -Seconds 45
}
if ($s -cmatch 'KernelWorkerStatus\.COMPLETE') {
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    kaggle kernels output $slug -p $out --force 2>&1
    Write-Output "OUTPUT_DOWNLOADED $out"
} else {
    Write-Output "WATCH_ENDED_WITHOUT_COMPLETE"
}
