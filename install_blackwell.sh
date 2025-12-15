#!/bin/bash

# SimKO Installation Script for Blackwell RTX 6000 GPUs
# CUDA 12.9 | PyTorch 2.8.0 cu129 | vLLM with Blackwell Support
# Configuration: 2 GPUs (GPU 0 and GPU 1)

set -e  # Exit on error

echo "=========================================="
echo "SimKO Blackwell Installation Script"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check CUDA version
echo "Checking CUDA version..."
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
echo "Detected CUDA version: $CUDA_VERSION"
if [[ ! "$CUDA_VERSION" =~ ^12\.[89] ]]; then
    echo "WARNING: CUDA version $CUDA_VERSION detected. This script is optimized for CUDA 12.8 or 12.9"
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Step 1: Creating conda environment 'verl_simko'..."
conda create -y -n verl_simko python=3.10.14

echo ""
echo "Step 2: Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl_simko

echo ""
echo "Step 3: Verifying Python version..."
python --version

echo ""
echo "Step 4: Installing PyTorch 2.8.0 with CUDA 12.9..."
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129

echo ""
echo "Step 5: Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"

echo ""
echo "Step 6: Installing verl package (without vllm)..."
pip install -e .

echo ""
echo "Step 7: Installing vLLM with CUDA 12.9 support..."
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129

echo ""
echo "Step 8: Installing additional requirements..."
pip install latex2sympy2 fire tensordict==0.7.2

echo ""
echo "Step 9: Verifying vLLM installation..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

echo ""
echo "Step 10: Testing GPU availability..."
python -c "import torch; print('GPU 0:', torch.cuda.get_device_name(0)); print('GPU 1:', torch.cuda.get_device_name(1))"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Environment: verl_simko"
echo "Python: 3.10.14"
echo "PyTorch: 2.8.0+cu129 (nightly)"
echo "vLLM: Latest with CUDA 12.9 support"
echo "GPUs configured: 2 (GPU 0 and GPU 1)"
echo ""
echo "IMPORTANT NOTES:"
echo "1. Flash Attention is NOT installed (not compatible with Blackwell)"
echo "2. Training scripts have been configured for 2 GPUs"
echo "3. Eager attention mode will be used (slightly slower but compatible)"
echo ""
echo "Next steps:"
echo "1. Activate environment: conda activate verl_simko"
echo "2. Run training: bash run_qwen2.5-math-7b_SimKO_2gpu.sh"
echo ""
