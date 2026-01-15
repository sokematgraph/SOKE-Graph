# PowerShell script for running SOKEGraph in Docker on Windows

Write-Host ""
Write-Host "SOKEGraph Docker Setup" -ForegroundColor Blue
Write-Host ""

# Create necessary directories on host
Write-Host "Creating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data/uploads", "data/outputs", "data/logs", "external/output" | Out-Null

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "No .env found." -ForegroundColor Yellow

    if (Test-Path ".env.example") {
        Write-Host "Creating .env from .env.example..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host "[OK] .env created from template." -ForegroundColor Green
    } else {
        Write-Host "[WARNING] No .env.example found. Creating a blank .env file..." -ForegroundColor Yellow
        New-Item -ItemType File -Path ".env" -Force | Out-Null
        Write-Host "[OK] Blank .env created." -ForegroundColor Green
    }
}

# Detect docker compose command (V2 vs V1)
Write-Host ""
$dockerComposeCmd = $null
try {
    $null = docker compose version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $dockerComposeCmd = "docker", "compose"
        Write-Host "Using Docker Compose V2 (docker compose)" -ForegroundColor Cyan
    }
} catch {
    # Docker Compose V2 not available
}

if (-not $dockerComposeCmd) {
    try {
        $null = docker-compose version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $dockerComposeCmd = "docker-compose"
            Write-Host "Using Docker Compose V1 (docker-compose)" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "[ERROR] Neither 'docker compose' nor 'docker-compose' found." -ForegroundColor Red
        Write-Host "Please install Docker Desktop or Docker Compose." -ForegroundColor Yellow
        exit 1
    }
}

if (-not $dockerComposeCmd) {
    Write-Host "[ERROR] Neither 'docker compose' nor 'docker-compose' found." -ForegroundColor Red
    Write-Host "Please install Docker Desktop or Docker Compose." -ForegroundColor Yellow
    exit 1
}

# Build and run
Write-Host ""
Write-Host "Building Docker image..." -ForegroundColor Yellow
if ($dockerComposeCmd -is [array]) {
    & $dockerComposeCmd[0] $dockerComposeCmd[1] build
} else {
    & $dockerComposeCmd build
}

Write-Host ""
Write-Host "Starting SOKEGraph application..." -ForegroundColor Yellow
if ($dockerComposeCmd -is [array]) {
    & $dockerComposeCmd[0] $dockerComposeCmd[1] up -d
} else {
    & $dockerComposeCmd up -d
}

# Wait for service to be ready
Write-Host ""
Write-Host "Waiting for application to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if container is running
$container = docker ps -q -f name=sokegraph-streamlit
if ($container) {
    Write-Host ""
    Write-Host "[SUCCESS] SOKEGraph is running!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access the application at: " -NoNewline -ForegroundColor Blue
    Write-Host "http://localhost:8501" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Your output files will be available in:" -ForegroundColor Blue
    Write-Host "  - ./data/outputs/       (ranking results, graphs)"
    Write-Host "  - ./external/output/    (pipeline outputs)"
    Write-Host "  - ./data/logs/          (application logs)"
    Write-Host ""
    Write-Host "Useful commands:" -ForegroundColor Blue
    if ($dockerComposeCmd -is [array]) {
        $cmdStr = "$($dockerComposeCmd[0]) $($dockerComposeCmd[1])"
    } else {
        $cmdStr = $dockerComposeCmd
    }
    Write-Host "  View logs:    $cmdStr logs -f"
    Write-Host "  Stop:         $cmdStr down"
    Write-Host "  Restart:      $cmdStr restart"
    Write-Host "  Shell access: docker exec -it sokegraph-streamlit bash"
} else {
    Write-Host ""
    Write-Host "[WARNING] Container failed to start. Check logs:" -ForegroundColor Yellow
    if ($dockerComposeCmd -is [array]) {
        & $dockerComposeCmd[0] $dockerComposeCmd[1] logs
    } else {
        & $dockerComposeCmd logs
    }
}