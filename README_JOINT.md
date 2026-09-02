# 联合训练流程使用说明

## 概述

联合训练流程将 Audio2Lip 和 Generator 进行端到端训练，让两个模型协同学习，在小批量专人数据集上训练出能还原人物口型特征的模型。

## 核心优势

1. **端到端优化**：Audio2Lip 和 Generator 协同学习，互相促进
2. **更合理的损失函数**：针对口型还原设计的多任务损失
3. **独立训练流程**：不影响现有的两个训练流程
4. **易于测试**：可以快速对比不同配置的效果

## 文件说明

- `train_joint.py` - 联合训练脚本
- `demo_joint.py` - 联合训练推理脚本
- `README_JOINT.md` - 本说明文档

## 训练流程

### 1. 数据准备

确保数据集结构如下：
```
wy/
├── lmdb/              # 图像LMDB数据库
├── lmdb_features/     # 特征LMDB数据库
├── mel/               # 音频mel特征 (.npy)
├── lip_feature/       # 唇部特征 (.npy)
├── pose_feature/      # 姿态特征 (.npy)
└── bbox/              # 边界框 (.npy)
```

### 2. 开始训练

```bash
python train_joint.py \
    --data_path wy \
    --epoch 20 \
    --batch_size 8 \
    --lr 0.0002 \
    --preload_features \
    --exp_name joint_v1
```

### 3. 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | wy | 数据集路径 |
| `--epoch` | 20 | 训练轮数 |
| `--batch_size` | 8 | 批量大小 |
| `--lr` | 0.0002 | 学习率 |
| `--preload_features` | False | 预加载特征到内存 |
| `--exp_name` | joint_v1 | 实验名称 |
| `--save_freq` | 5 | 保存频率 |
| `--resume_ckpt` | None | 恢复检查点路径 |

### 4. 查看训练日志

```bash
tensorboard --logdir logs/joint_training
```

## 推理流程

### 1. 使用训练后的模型推理

```bash
python demo_joint.py \
    --ckpt_path logs/joint_training/joint_v1/.../checkpoint/joint_epoch_0019_iter_050000.pt \
    --source_path test_data/output.mp4 \
    --audio_driving_path test_data/wy5.wav \
    --save_path res/joint_output.mp4
```

### 2. 推理参数说明

| 参数 | 说明 |
|------|------|
| `--ckpt_path` | 联合训练检查点路径（必需） |
| `--source_path` | 源视频路径 |
| `--audio_driving_path` | 驱动音频路径 |
| `--save_path` | 输出视频路径 |
| `--need_crop_pose_video` | 是否裁剪视频 |

## 损失函数说明

联合训练使用多任务损失：

| 损失项 | 权重 | 作用 |
|--------|------|------|
| 唇部特征损失 (lip_loss) | 10 | Audio2Lip 预测准确性 |
| 唇部平滑损失 (lip_smooth_loss) | 5 | 唇部特征时序平滑性 |
| 图像重建损失 (img_l1_loss) | 1 | 像素级重建质量 |
| 图像平滑损失 (img_smooth_loss) | 1 | 图像时序平滑性 |
| 嘴部区域损失 (mouth_l1_loss) | 5 | 嘴部区域重建质量 |

## 模型架构

联合训练的模型架构：

```
音频 → Audio2Lip → 唇部特征预测 (20维)
                           ↓
源图像 → Generator.Encoder → 身份特征 + 背景特征
                           ↓
唇部特征 + 姿态特征 → Generator.Decoder → 生成图像
```

## 与现有流程的对比

| 方面 | 现有流程 | 联合训练 |
|------|----------|----------|
| 训练方式 | 两个模型独立训练 | 端到端联合训练 |
| 模型协同 | 无 | Audio2Lip 和 Generator 协同学习 |
| 损失函数 | 单一任务损失 | 多任务联合损失 |
| 推理方式 | 手动组合两个模型权重 | 单一检查点包含所有权重 |
| 适用场景 | 大数据集 | 小批量专人数据集 |

## 优化建议

### 1. 调整学习率

- 如果训练不稳定，降低学习率：`--lr 0.0001`
- 如果收敛太慢，提高学习率：`--lr 0.0003`

### 2. 调整批量大小

- 显存充足时增大批量：`--batch_size 16`
- 显存不足时减小批量：`--batch_size 4`

### 3. 调整训练轮数

- 小数据集建议训练更多轮：`--epoch 30`
- 大数据集可以减少轮数：`--epoch 10`

### 4. 使用预加载特征

- 小数据集建议启用：`--preload_features`
- 大数据集不建议使用（显存占用大）

## 常见问题

### Q1: 训练速度慢怎么办？

A: 
- 增大批量大小：`--batch_size 16`
- 启用预加载特征：`--preload_features`
- 检查 GPU 使用率

### Q2: 训练不稳定怎么办？

A:
- 降低学习率：`--lr 0.0001`
- 减小批量大小：`--batch_size 4`
- 检查数据质量

### Q3: 推理效果不好怎么办？

A:
- 增加训练轮数
- 调整损失权重（修改 train_joint.py）
- 检查训练数据质量

### Q4: 如何恢复训练？

A:
```bash
python train_joint.py \
    --resume_ckpt logs/joint_training/joint_v1/.../checkpoint/joint_epoch_0009_iter_025000.pt \
    --data_path wy \
    --epoch 30
```

## 注意事项

1. **数据质量**：确保训练数据质量良好，口型清晰
2. **显存管理**：联合训练显存占用较大，建议使用 16GB 以上显存
3. **训练时间**：联合训练时间较长，建议预留充足时间
4. **备份检查点**：定期备份检查点，防止训练中断丢失进度

## 技术支持

如有问题，请查看：
1. 训练日志：`logs/joint_training/`
2. TensorBoard：`tensorboard --logdir logs/joint_training`
3. 检查点文件：确认文件完整性
