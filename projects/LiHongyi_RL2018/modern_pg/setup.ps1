$ErrorActionPreference = 'Stop'
# 只创建新环境；已有同名环境时不安装、不覆盖。
$envName = 'mlds2018-hw4'
$envList = conda env list --json | ConvertFrom-Json
if ($envList.envs | Where-Object { (Split-Path $_ -Leaf) -eq $envName }) {
    throw "环境 $envName 已存在。为避免覆盖，请使用现有环境或另选名称。"
}
$env:CONDA_PKGS_DIRS = Join-Path $env:USERPROFILE '.conda/pkgs-mlds2018'
conda create -n $envName python=3.11.17 pip -y --override-channels -c https://repo.anaconda.com/pkgs/main
if ($LASTEXITCODE -ne 0) { throw 'Conda 环境创建失败' }
$envList = conda env list --json | ConvertFrom-Json
$envPath = $envList.envs | Where-Object { (Split-Path $_ -Leaf) -eq $envName } | Select-Object -First 1
$envPython = Join-Path $envPath 'python.exe'
& $envPython -m pip install -r (Join-Path $PSScriptRoot 'requirements-lock.txt')
if ($LASTEXITCODE -ne 0) { throw '依赖安装失败' }
& $envPython -m pip check
if ($LASTEXITCODE -ne 0) { throw '依赖检查失败' }
& $envPython -B (Join-Path $PSScriptRoot 'smoke_test.py') --output (Join-Path $PSScriptRoot 'smoke_results-reproduced.json')
if ($LASTEXITCODE -ne 0) { throw 'Smoke Test 失败，查看 JSON 报告' }
