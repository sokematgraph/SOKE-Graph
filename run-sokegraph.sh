#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# SOKEGraph - Automated Setup and Launch Script (macOS/Linux)
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="sokegraph"
PYTHON_VERSION="3.10"

echo -e "${BLUE}"
echo "═══════════════════════════════════════════════════════════════"
echo "   SOKEGraph - Research Paper Knowledge Graph Builder"
echo "═══════════════════════════════════════════════════════════════"
echo -e "${NC}"

# ─────────────────────────────────────────────────────────────────
# Step 1: Check if Conda is installed
# ─────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/4] Checking for Conda installation...${NC}"

if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ ERROR: Conda is not installed or not in PATH${NC}"
    echo ""
    echo "Please install Miniconda or Anaconda from:"
    echo "  • Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo "  • Anaconda:  https://www.anaconda.com/download"
    echo ""
    echo "After installation, restart your terminal and run this script again."
    exit 1
fi

echo -e "${GREEN}✓ Conda found: $(conda --version)${NC}"

# ─────────────────────────────────────────────────────────────────
# Step 2: Create/activate conda environment
# ─────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[2/4] Setting up conda environment '${ENV_NAME}'...${NC}"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${GREEN}✓ Environment '${ENV_NAME}' already exists${NC}"
else
    echo "Creating new conda environment with Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    echo -e "${GREEN}✓ Environment created successfully${NC}"
fi

# Activate the environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

if [[ "$CONDA_DEFAULT_ENV" != "${ENV_NAME}" ]]; then
    echo -e "${RED}✗ ERROR: Failed to activate conda environment${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Environment '${ENV_NAME}' activated${NC}"

# ─────────────────────────────────────────────────────────────────
# Step 3: Install dependencies
# ─────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[3/4] Installing dependencies from requirements.txt...${NC}"

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}✗ ERROR: requirements.txt not found in current directory${NC}"
    exit 1
fi

pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed successfully${NC}"

# ─────────────────────────────────────────────────────────────────
# Step 4: Launch Streamlit application
# ─────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[4/4] Starting SOKEGraph Streamlit application...${NC}"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  🚀 SOKEGraph is starting...${NC}"
echo -e "${GREEN}  📍 The application will open in your default browser${NC}"
echo -e "${GREEN}  🌐 Default URL: http://localhost:8501${NC}"
echo ""
echo -e "${GREEN}  Press Ctrl+C to stop the application${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Run streamlit
streamlit run streamlit-app.py

# This line will execute when the user stops streamlit
echo ""
echo -e "${BLUE}SOKEGraph application stopped.${NC}"
