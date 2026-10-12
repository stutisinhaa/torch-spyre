#!/bin/bash
# Copyright 2025 The Torch-Spyre Authors.
# Build script for torch-spyre with stub dependencies (Python-only, no C++ compilation)

set -e

echo "=========================================="
echo "Building torch-spyre wheel with stubs"
echo "Python-only build (no C++ compilation)"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Define mockdevice path - can be overridden via environment variable or command line
MOCKDEVICE_DIR="${MOCKDEVICE_DIR}"
echo MOCKDEVICE_DIR: $MOCKDEVICE_DIR
MOCKDEVICE_WHEEL="$MOCKDEVICE_DIR/dist/mockdevice-0.1.0-py3-none-any.whl"

echo "Using mock-device directory: $MOCKDEVICE_DIR"

# Step 1: Uninstall existing packages
echo ""
echo "Step 1: Uninstalling existing packages..."
echo "=========================================="
python3 -m pip uninstall torch_spyre -y 2>/dev/null || echo "torch_spyre not installed"
python3 -m pip uninstall mockdevice -y 2>/dev/null || echo "mockdevice not installed"

# Step 2: Always rebuild mockdevice wheel to ensure latest changes
echo ""
echo "Step 2: Building mockdevice wheel..."
echo "=========================================="
echo "Rebuilding mockdevice to ensure latest changes are included..."
cd "$MOCKDEVICE_DIR"

# Clean previous mockdevice builds to force rebuild
echo "Cleaning previous mockdevice builds..."
rm -rf dist/ build/ *.egg-info 2>/dev/null || true

# Build mockdevice wheel
./build_wheel.sh

cd "$SCRIPT_DIR"

# Verify the wheel was created
if [ ! -f "$MOCKDEVICE_WHEEL" ]; then
    echo "ERROR: Failed to build mockdevice wheel at $MOCKDEVICE_WHEEL"
    exit 1
fi
echo "mockdevice wheel built successfully: $MOCKDEVICE_WHEEL"

# Step 3: Install mockdevice wheel
echo ""
echo "Step 3: Installing mockdevice wheel..."
echo "=========================================="
python3 -m pip install --no-deps "$MOCKDEVICE_WHEEL"

# Step 4: Build torch-spyre wheel
echo ""
echo "Step 4: Building torch-spyre wheel..."
echo "=========================================="

# Enable stub mode - this skips C++ compilation
export USE_STUBS=1

echo "Environment setup:"
echo "  USE_STUBS=$USE_STUBS (C++ compilation disabled)"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.egg-info
python3 setup.py clean 2>/dev/null || true

# Build the wheel using setup.py with USE_STUBS=1
echo ""
echo "Building torch-spyre wheel with USE_STUBS=1..."
python3 setup.py bdist_wheel

# Step 5: Install torch-spyre wheel
echo ""
echo "Step 5: Installing torch-spyre wheel..."
echo "=========================================="
TORCH_SPYRE_WHEEL=$(ls dist/torch_spyre-*.whl 2>/dev/null | head -n 1)
if [ -z "$TORCH_SPYRE_WHEEL" ]; then
    echo "ERROR: torch-spyre wheel not found in dist/"
    exit 1
fi
echo "Installing: $TORCH_SPYRE_WHEEL"
python3 -m pip install --no-deps "$TORCH_SPYRE_WHEEL"

echo ""
echo "=========================================="
echo "Build and Installation Complete!"
echo "=========================================="
echo ""
echo "Installed packages:"
echo "  - mockdevice: $(python3 -m pip show mockdevice 2>/dev/null | grep Version | cut -d' ' -f2)"
echo "  - torch-spyre: $(python3 -m pip show torch-spyre 2>/dev/null | grep Version | cut -d' ' -f2)"
echo ""
echo "Note: This is a Python-only build using mock device."
echo "No C++ extensions are compiled. The mock device enables:"
echo "  - PyTorch Inductor integration"
echo "  - SDSC JSON generation"
echo "  - CPU-based execution for testing"
