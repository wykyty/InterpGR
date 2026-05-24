# T5 Decoder Hook Points

TransformerLens 为 T5 的每个 decoder 层暴露了多个 hook point，可以用来提取中间表示或做干预实验。

## 每层计算流程

```
resid_pre（上一层输出 / embedding）
    │
    ├──→ self_attn（自注意力，decoder token 之间互看）
    │
    ▼
resid_mid = resid_pre + attn_out
    │
    ├──→ cross_attn（交叉注意力，decoder 去看 encoder 输出）
    │
    ▼
resid_mid_cross = resid_mid + cross_attn_out
    │
    ├──→ MLP
    │
    ▼
resid_post = resid_mid_cross + mlp_out
```

## Hook Point 说明

| Hook Name | 维度 | 含义 |
|---|---|---|
| `hook_resid_pre` | `[batch, pos, d_model]` | 该层输入，即上一层的残差流输出 |
| `hook_attn_out` | `[batch, pos, d_model]` | **自注意力**输出（decoder token 之间的注意力） |
| `hook_resid_mid` | `[batch, pos, d_model]` | 自注意力残差连接后的结果：`resid_pre + attn_out` |
| `hook_cross_attn_out` | `[batch, pos, d_model]` | **交叉注意力**输出（decoder 对 encoder 的注意力） |
| `hook_resid_mid_cross` | `[batch, pos, d_model]` | 交叉注意力残差连接后的结果：`resid_mid + cross_attn_out` |
| `hook_mlp_out` | `[batch, pos, d_model]` | MLP 输出 |
| `hook_resid_post` | `[batch, pos, d_model]` | 该层最终输出：`resid_mid_cross + mlp_out`，也是下一层输入 |

## 自注意力 vs 交叉注意力

- **自注意力** (`hook_attn_out`)：Q/K/V 都来自 decoder 自身的残差流（经 LayerNorm），decoder token 之间互相看。带 causal mask，只能看到当前位置之前的 token。
- **交叉注意力** (`hook_cross_attn_out`)：Q 来自 decoder 残差流（经 LayerNorm），K/V 来自 encoder 最终输出。decoder 每层都能看到完整的 encoder 表示，用于提取源端信息。

注意：所有 decoder 层共享同一个 encoder 输出作为 K/V，不会对 encoder 输出再做逐层的 LayerNorm。

## 残差流（Residual Stream）

T5 使用 Pre-LayerNorm（RMSNorm）架构。每个子模块不是替换表示，而是把输出**累加**到残差流上：

```
resid_post = resid_mid_cross + mlp_out
           = (resid_mid + cross_attn_out) + mlp_out
           = ((resid_pre + attn_out) + cross_attn_out) + mlp_out
```

所以 `hook_resid_post` 包含了自注意力、交叉注意力、MLP 三者的贡献。

## SAE 训练选哪个 Hook

| Hook | 研究什么 | 推荐场景 |
|---|---|---|
| `decoder.{L}.hook_mlp_out` | MLP 学到的模式 | 默认选择，干扰少，模式清晰 |
| `decoder.{L}.hook_cross_attn_out` | decoder 从 encoder 提取的信息 | 研究编解码信息流动 |
| `decoder.{L}.hook_resid_post` | 整层综合表示 | 最全面但也最混杂 |
| `decoder.{L}.hook_resid_mid` | 自注意力后的表示 | 研究 decoder 自身的注意力模式 |

## Hook 命名规则

```
decoder.{layer}.{hook_name}
encoder.{layer}.{hook_name}
```

T5-large 有 24 层（0-23），`d_model=1024`。所有 hook point 的输出维度都是 `[batch, seq, 1024]`，所以切换 hook 只需改 `hook_name`，不需要改 `d_in`。
