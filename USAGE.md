# QWEN2.5 GRPO 300 Steps 实验恢复指南

## 实验结果
- **最终验证指标**: val-core/econ_agent/reward/mean@1 = 0.0988
- **训练步数**: 300 steps
- **训练时间**: 4小时15分钟

## 仓库内容

```
├── checkpoints/global_step_300/actor/
│   ├── lora_adapter/          # LoRA权重 (78MB) - 最重要！
│   └── huggingface/           # 模型配置和tokenizer
├── config/
│   └── econ_grpo_small.yaml   # 训练配置
├── RL/
│   ├── reward.py              # 奖励函数
│   └── prepare_verl_data.py   # 数据准备脚本
├── ai_economist/              # AI经济学仿真环境
├── data/
│   ├── verl_dataset_small/    # 训练/验证数据集
│   ├── profiles.json          # Agent档案
│   └── gpt-3.../good_decisions.csv  # 好决策样本
├── simulate.py                # 仿真脚本
├── train_300steps.log         # 完整训练日志
└── requirements.txt           # 依赖
```

## 新实例恢复步骤

### 1. 克隆仓库
```bash
cd /workspace
git clone https://github.com/uqcxu9/QWEN2.5_MACRO_300_PRE.git
cd QWEN2.5_MACRO_300_PRE
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
pip install -r GRPO_requirements.txt
```

### 3. 下载基础模型
```bash
# 需要下载 Qwen2.5-7B-Instruct 作为基础模型
# 可以使用 huggingface-cli 或手动下载到 /workspace/models/Qwen2.5-7B-Instruct
```

### 4. 加载LoRA模型运行仿真
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "/workspace/models/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# 加载LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "checkpoints/global_step_300/actor/lora_adapter"
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "checkpoints/global_step_300/actor/huggingface"
)

# 现在可以使用 model 进行推理
```

### 5. 运行仿真实验
```bash
python simulate.py --help
```

## 继续训练（从300步继续）

如果需要继续训练到700步：
1. 修改 `config/econ_grpo_small.yaml` 中的 `total_training_steps: 700`
2. 设置 `resume_mode: auto`
3. 运行训练脚本

## 重要文件说明

| 文件 | 说明 |
|------|------|
| `lora_adapter/adapter_model.safetensors` | LoRA权重 (78MB) |
| `train_300steps.log` | 完整训练日志，包含所有验证分数 |
| `data/verl_dataset_small/` | 训练和验证数据集 |
| `config/econ_grpo_small.yaml` | 训练超参数配置 |

## 注意事项

1. 完整的模型权重 (15GB) 未上传，仅上传了LoRA adapter
2. 需要基础模型 Qwen2.5-7B-Instruct 配合使用
3. 路径已改为相对路径，直接在仓库目录下运行即可
