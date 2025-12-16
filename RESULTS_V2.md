# QWEN2.5 GRPO V2 训练结果 (新 reward.py)

## 训练配置
- 基础模型: Qwen2.5-7B-Instruct
- 训练步数: 300 steps
- LoRA rank: 8
- 训练时间: 约4小时

## 验证集结果对比

| 指标 | V1 (旧reward) | V2 (新reward) | 变化 |
|------|---------------|---------------|------|
| **验证集Mean Reward** | 0.0988 | **0.1808** | **+83%** |
| 正分数占比 | 86.0% | 82.0% | -4% |

## 模型行为改进

### Work 分布
| 版本 | 主要值 | 说明 |
|------|--------|------|
| V1 | work=1.0 (77.5%) | 锁死在满工作 |
| V2 | work=0.8 (79.5%), work=0.6 (18.5%) | ✅ 多样化 |

### Consumption 分布
| 版本 | 主要值 | 说明 |
|------|--------|------|
| V1 | consumption=0.60 (77.5%) | 单一 |
| V2 | consumption=0.70 (62%), consumption=0.60 (32.5%) | ✅ 更合理 |

## 新 reward.py 关键改动

1. **Work行为使用区间奖励**：避免work=1成为"保险动作"
2. **过劳惩罚**：work > 0.95 时施加惩罚
3. **过度消费惩罚**：consumption > 0.90 时施加惩罚
4. **Buffer Ratio分档**：
   - buffer_ratio < 2: 期望 work ∈ [0.70, 0.90]
   - buffer_ratio 2-4: 期望 work ∈ [0.35, 0.75]
   - buffer_ratio > 4: 期望 work ∈ [0.10, 0.35]

## 文件说明

- `RL/reward.py` - 更新后的奖励函数
- `train_v2_300steps.log` - V2版本训练日志
- `checkpoints_v2/` - V2版本LoRA权重

## 加载V2模型

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    "/workspace/models/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "checkpoints_v2/global_step_300/actor/lora_adapter"
)
```
