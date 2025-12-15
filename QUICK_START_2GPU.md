# SimKO Quick Start - 2 GPU Blackwell Setup

## TL;DR - Installation Commands

```bash
# Navigate to SimKO directory
cd ~/arp/sim_implementation/SimKO

# Run automated installation
bash install_blackwell.sh

# Activate environment
conda activate verl_simko

# Run training (choose one)
bash run_qwen2.5-math-7b_SimKO_2gpu.sh       # SimKO method
bash run_qwen2.5-math-7b_grpo_2gpu.sh        # GRPO baseline
bash run_llama3.2-3b_SimKO_2gpu.sh           # Llama-3.2-3B SimKO
```

## What Got Changed?

### 1. Files Modified
- ✅ `setup.py` - Removed vllm version constraint, removed flash-attn
- ✅ `requirements.txt` - Updated for Blackwell compatibility

### 2. New Training Scripts Created
- ✅ `run_qwen2.5-math-7b_SimKO_2gpu.sh` - 2-GPU SimKO training
- ✅ `run_qwen2.5-math-7b_grpo_2gpu.sh` - 2-GPU GRPO baseline
- ✅ `run_llama3.2-3b_SimKO_2gpu.sh` - 2-GPU Llama SimKO training

### 3. Key Configuration Changes
- GPUs: 8 → **2** (GPU 0 and GPU 1)
- Batch size: 1024 → **512**
- Mini batch: 256 → **128**
- Flash Attention: **DISABLED** (enforce_eager=True)
- GPU Memory: **0.65-0.75** (optimized for 98GB VRAM)

## Installation Packages

```
✅ Python 3.10.14
✅ PyTorch 2.8.0+cu129 (nightly)
✅ vLLM latest (CUDA 12.9 support)
✅ tensordict 0.7.2
✅ latex2sympy2
✅ fire
❌ flash-attn (NOT compatible with Blackwell)
```

## Verification Checklist

After installation, verify:

```bash
# 1. Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"  # Should be True

# 2. Check GPU count
python -c "import torch; print(torch.cuda.device_count())"  # Should be 2

# 3. Check vLLM
python -c "import vllm; print(vllm.__version__)"  # Should show version

# 4. Check GPU names
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(2)]"
```

## Critical Notes

⚠️ **Flash Attention**: Disabled - not compatible with Blackwell architecture
⚠️ **GPU Selection**: Only GPU 0 and GPU 1 (GPU 3 is in use by another process)
⚠️ **CUDA Version**: Using cu129 (CUDA 12.9) for native Blackwell support
✅ **Eager Attention**: ~10-15% slower but fully compatible

## File Structure

```
SimKO/
├── install_blackwell.sh                    # ← Automated installation script
├── run_qwen2.5-math-7b_SimKO_2gpu.sh      # ← NEW: 2-GPU SimKO training
├── run_qwen2.5-math-7b_grpo_2gpu.sh       # ← NEW: 2-GPU GRPO baseline
├── run_llama3.2-3b_SimKO_2gpu.sh          # ← NEW: 2-GPU Llama training
├── BLACKWELL_SETUP.md                      # ← Complete setup guide
├── QUICK_START_2GPU.md                     # ← This file
├── setup.py                                # ← MODIFIED for Blackwell
├── requirements.txt                        # ← MODIFIED for Blackwell
└── data/                                   # Data directory
    ├── math/
    ├── aime2025/
    ├── amc23/
    └── gsm8k_level1/
```

## Monitoring Training

```bash
# Terminal 1: Run training
conda activate verl_simko
bash run_qwen2.5-math-7b_SimKO_2gpu.sh

# Terminal 2: Monitor GPUs
watch -n 1 nvidia-smi
```

## Common Issues & Quick Fixes

**Q: Out of memory error?**
A: Reduce batch size to 256 in the training script

**Q: "Flash attention" warnings?**
A: Normal - it's disabled for Blackwell compatibility

**Q: vLLM errors?**
A: Update vLLM: `pip install --upgrade vllm --extra-index-url https://download.pytorch.org/whl/cu129`

**Q: Wrong GPU being used?**
A: Scripts set `CUDA_VISIBLE_DEVICES=0,1` automatically

---

For detailed information, see **BLACKWELL_SETUP.md**
