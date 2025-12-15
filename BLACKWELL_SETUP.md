# SimKO Setup Guide for Blackwell RTX 6000 GPUs

## System Configuration
- **GPUs**: 2x NVIDIA RTX PRO 6000 Blackwell Server Edition (98GB VRAM each)
- **CUDA**: 12.9
- **Driver**: 580.95.05
- **Architecture**: Blackwell (Compute Capability 12.2)

## Overview

This guide provides step-by-step instructions for setting up SimKO on Blackwell RTX 6000 GPUs with proper compatibility for CUDA 12.9.

### Key Compatibility Notes

1. **Flash Attention**: NOT compatible with Blackwell architecture - we use eager attention mode instead
2. **vLLM**: Using latest version with CUDA 12.9 (cu129) support
3. **PyTorch**: Using PyTorch 2.8.0 nightly with cu129 for native Blackwell support
4. **GPU Configuration**: Modified for 2 GPUs (GPU 0 and GPU 1)

---

## Step-by-Step Installation

### **STEP 1: Navigate to SimKO Directory**

```bash
cd ~/arp/sim_implementation/SimKO
```

### **STEP 2: Run Automated Installation Script**

```bash
# Make the script executable (if not already done)
chmod +x install_blackwell.sh

# Run the installation script
bash install_blackwell.sh
```

The script will automatically:
- Create conda environment `verl_simko` with Python 3.10.14
- Install PyTorch 2.8.0 with CUDA 12.9 support
- Install vLLM with Blackwell compatibility
- Install all required dependencies
- Verify the installation

**Expected Duration**: 10-15 minutes

---

### **Alternative: Manual Installation**

If you prefer manual installation, follow these steps:

#### Step 2.1: Create Conda Environment

```bash
conda create -y -n verl_simko python=3.10.14
conda activate verl_simko
```

#### Step 2.2: Install PyTorch with CUDA 12.9

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
```

#### Step 2.3: Verify PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"
```

**Expected Output:**
```
PyTorch version: 2.8.0+cu129 (or similar)
CUDA available: True
CUDA version: 12.9
Number of GPUs: 2
```

#### Step 2.4: Install verl Package

```bash
pip install -e .
```

#### Step 2.5: Install vLLM with CUDA 12.9

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
```

#### Step 2.6: Install Additional Requirements

```bash
pip install latex2sympy2 fire tensordict==0.7.2
```

#### Step 2.7: Verify Installation

```bash
# Verify vLLM
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Test GPU accessibility
python -c "import torch; print('GPU 0:', torch.cuda.get_device_name(0)); print('GPU 1:', torch.cuda.get_device_name(1))"
```

---

## Modified Files Summary

### 1. **setup.py**
- Removed `vllm<=0.6.3` constraint (commented out)
- Removed `flash-attn` from GPU_REQUIRES (not compatible with Blackwell)

### 2. **requirements.txt**
- Commented out `vllm==0.8.2` (will use latest version)
- Added note about flash-attn incompatibility

### 3. **Training Scripts (New 2-GPU versions)**

#### Created Files:
- `run_qwen2.5-math-7b_SimKO_2gpu.sh` - SimKO training for Qwen2.5-Math-7B
- `run_qwen2.5-math-7b_grpo_2gpu.sh` - GRPO baseline for Qwen2.5-Math-7B
- `run_llama3.2-3b_SimKO_2gpu.sh` - SimKO training for Llama-3.2-3B

#### Key Modifications:
- `CUDA_VISIBLE_DEVICES=0,1` - Use only GPU 0 and GPU 1
- `trainer.n_gpus_per_node=2` - Changed from 8 to 2
- `data.train_batch_size=512` - Reduced from 1024 (adjusted for 2 GPUs)
- `actor.ppo_mini_batch_size=128` - Reduced from 256 (adjusted for 2 GPUs)
- `rollout.enforce_eager=True` - **CRITICAL**: Disable flash attention
- `rollout.gpu_memory_utilization=0.65-0.75` - Optimized for 98GB VRAM
- Updated experiment names to reflect 2-GPU Blackwell configuration

---

## Running Training

### **Option 1: SimKO Training (Qwen2.5-Math-7B)**

```bash
conda activate verl_simko
bash run_qwen2.5-math-7b_SimKO_2gpu.sh
```

### **Option 2: GRPO Baseline (Qwen2.5-Math-7B)**

```bash
conda activate verl_simko
bash run_qwen2.5-math-7b_grpo_2gpu.sh
```

### **Option 3: SimKO Training (Llama-3.2-3B)**

```bash
conda activate verl_simko
bash run_llama3.2-3b_SimKO_2gpu.sh
```

---

## Verification Steps

### Check GPU Usage During Training

```bash
# In another terminal
watch -n 1 nvidia-smi
```

You should see:
- GPU 0 and GPU 1 active
- Memory usage around 60-75GB per GPU during training
- GPU utilization ~80-95%

### Monitor Training Logs

The training will log to:
- **Console**: Real-time progress
- **Weights & Biases**: Online dashboard (if configured)

---

## Troubleshooting

### Issue 1: "CUDA out of memory"

**Solution**: Reduce batch size or memory utilization

Edit the training script:
```bash
# Reduce from 512 to 256
data.train_batch_size=256

# Or reduce memory utilization
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

### Issue 2: "Flash attention not available" warnings

**Expected**: This is normal - flash attention is not compatible with Blackwell. The system will automatically use eager attention mode.

### Issue 3: vLLM errors about compute capability

**Solution**: Ensure you have the latest vLLM version
```bash
pip install --upgrade vllm --extra-index-url https://download.pytorch.org/whl/cu129
```

### Issue 4: GPU 3 in use (conflict)

If GPU 3 is already in use by another process:
```bash
# The scripts are already configured to use only GPU 0 and 1
# Verify with:
echo $CUDA_VISIBLE_DEVICES  # Should show: 0,1
```

### Issue 5: ImportError or module not found

**Solution**: Reinstall in the correct environment
```bash
conda activate verl_simko
pip install -e .
```

---

## Performance Expectations

### Training Speed (2 GPUs vs 8 GPUs)

- **Original (8 GPUs)**: ~X samples/second
- **Modified (2 GPUs)**: ~X/4 samples/second (expected)
- **Eager Attention Overhead**: ~10-15% slower than flash attention

### Memory Usage

- **Per GPU**: 50-75GB during training
- **Batch Size**: Adjusted to 512 (from 1024)
- **Gradient Checkpointing**: Enabled to save memory

---

## Technical Details

### Why These Changes?

1. **Flash Attention Disabled**: Blackwell architecture (sm_120/sm_122) is not yet supported by Flash Attention 2 or 3

2. **Eager Attention**: Fully compatible with all GPU architectures, slightly slower but reliable

3. **CUDA 12.9 cu129**: Native support for Blackwell, better performance than compatibility mode

4. **2-GPU Configuration**: Optimized for your hardware setup while maintaining training quality

5. **Batch Size Reduction**: Scaled proportionally to maintain similar effective batch size per GPU

---

## References & Resources

- **vLLM Documentation**: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
- **PyTorch Blackwell Support**: https://pytorch.org/blog/pytorch-2-7/
- **NVIDIA Blackwell Guide**: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
- **Original SimKO Paper**: https://arxiv.org/abs/2510.14807
- **Flash Attention Blackwell Issue**: https://github.com/Dao-AILab/flash-attention/issues/1987

---

## Next Steps After Installation

1. **Verify Data Files**: Ensure datasets are present in `./data/` directory
2. **Configure Weights & Biases** (optional): Set up W&B for experiment tracking
3. **Start Training**: Run one of the training scripts
4. **Monitor Progress**: Check GPU usage and training metrics
5. **Evaluate Results**: Compare SimKO vs GRPO performance

---

## Questions or Issues?

If you encounter any problems:
1. Check the troubleshooting section above
2. Verify all installation steps completed successfully
3. Check the GitHub issues for the original SimKO repository
4. Review vLLM and PyTorch documentation for Blackwell-specific issues

---

**Installation created**: 2025-12-15
**Configuration**: 2x Blackwell RTX 6000, CUDA 12.9, PyTorch 2.8.0+cu129
