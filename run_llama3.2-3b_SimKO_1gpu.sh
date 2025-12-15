#!/bin/bash

# SimKO Training Script (Llama 3.2-3B) - Modified for 1 Blackwell RTX 6000 GPU
# CUDA 12.9 | PyTorch 2.8.0 cu129 | vLLM (Eager Attention Mode)
# GPU: 0

export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=0

math_train_path=./data/gsm8k_level1/train.parquet
math_test_path=./data/math/test.parquet
aime2025_test_path=./data/aime2025/test.parquet
amc23_test_path=./data/amc23/test.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path', '$aime2025_test_path', '$amc23_test_path']"
kl_coef=0
lr=1e-6
model_name=meta-llama/Llama-3.2-3B-Instruct

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.simko=True \
    actor_rollout_ref.actor.top_k=3 \
    actor_rollout_ref.actor.tau=0.8 \
    actor_rollout_ref.actor.mix_topk_coef=0.01 \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.experiment_name="MATH-Llama-3.2-3B-SimKO-1GPU-Blackwell" \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb','console'] \
    trainer.project_name='SimKO-Blackwell' \
    trainer.n_gpus_per_node=1 \
    +trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=8 \
    trainer.test_freq=8 \
    trainer.total_epochs=20 $@

# Key Changes for 1-GPU Blackwell Setup:
# 1. CUDA_VISIBLE_DEVICES=0 - Use only GPU 0
# 2. trainer.n_gpus_per_node=1 - Single GPU training
# 3. rollout.tensor_model_parallel_size=1 - No tensor parallelism (single GPU)
# 4. data.train_batch_size=128 - Reduced from 1024 (aggressive memory optimization)
# 5. data.max_response_length=1536 - Reduced from 3072 (aggressive memory optimization)
# 6. actor.ppo_mini_batch_size=32 - Reduced from 256 (aggressive memory optimization)
# 7. actor.ppo_max_token_len_per_gpu=4000 - Reduced from 12000 (aggressive memory optimization)
# 8. rollout.log_prob_max_token_len_per_gpu=4000 - Reduced from 12000 (aggressive memory optimization)
# 9. ref.log_prob_max_token_len_per_gpu=4000 - Reduced from 12000 (aggressive memory optimization)
# 10. rollout.enforce_eager=True - CRITICAL: Disable flash attention (not compatible with Blackwell)
# 11. rollout.gpu_memory_utilization=0.60 - Conservative vLLM memory allocation
# 12. param_offload=True and optimizer_offload=True - Offload to CPU to save GPU memory
# 13. trainer.experiment_name updated to reflect 1-GPU Blackwell config
#
# Note: Single GPU setup uses sleep/wake mechanism where FSDP training sleeps
# and releases GPU memory for vLLM inference, then wakes up for next training step.
# This is slower than 2-GPU tensor parallel but uses less memory and is simpler.
#
# Memory footprint per batch: ~128 samples × 1536 tokens = 196K tokens
# With Llama 3.2-3B (vocab ~128K), logits memory: 196K × 128K × 4 bytes ≈ 100 GB
# SimKO requires computing top-K log probs which increases memory pressure significantly.
