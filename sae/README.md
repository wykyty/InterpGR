```
checkpoints/batchtopk_sae_t5_large_decoder/
├── sae_weights.safetensors        # 训练权重（原始）
├── sae_config.json                # 训练配置
├── training_metrics.json
├── sparsity.safetensors
├── runner_cfg.json
└── inference/                     # 推理格式（SAELens 兼容）
    ├── sae_weights.safetensors    # JumpReLU 格式（topk_threshold → threshold）
    ├── cfg.json                   # SAELens 标准配置
    ├── scaling_factor.safetensors
    └── sparsity.safetensors

```