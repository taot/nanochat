# nanochat 分布式训练

## 进程启动

使用 `torchrun` 启动，每张 GPU 一个进程：

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train
```

初始化在 `nanochat/common.py` 的 `compute_init()` 中完成，使用 NCCL 后端：

```python
dist.init_process_group(backend="nccl", device_id=device)
```

---

## 数据分片

`nanochat/dataloader.py` 实现**步进式分片**，各 rank 独立加载，无需全局同步：

- Rank 0 处理 row group 0、N、2N、...
- Rank 1 处理 row group 1、N+1、2N+1、...

支持精确断点续训，保存 `(pq_idx, rg_idx, epoch)` 到 checkpoint。

---

## 梯度同步

`nanochat/optim.py` 实现自定义 `DistMuonAdamW` 优化器，**不使用 PyTorch DDP**，而是在 `optimizer.step()` 内手动管理 NCCL 通信，采用 ZeRO-2 风格的异步三阶段流程：

| 阶段 | 操作 |
|------|------|
| Phase 1 | 异步发起 reduce（小参数用 `all_reduce`，大参数用 `reduce_scatter`）|
| Phase 2 | 等待 reduce 完成，每个 rank 只更新自己负责的参数切片 |
| Phase 3 | `all_gather` 收集所有 rank 的更新，写回完整参数 |

**小参数**（< 1024 元素）：全量 all_reduce，优化器状态不分片。  
**大参数**：优化器状态按 rank 分片，每 rank 只维护 1/N 的状态。  
**Muon 参数组**：K 个参数堆叠后按 rank 分块，reduce_scatter → 本地 Muon 更新 → all_gather。

---

## Checkpoint

`nanochat/checkpoint_manager.py`：

| 内容 | 策略 |
|------|------|
| 模型权重 | 只有 rank 0 保存（`model_{step}.pt`）|
| 优化器状态 | 每个 rank 各自保存自己的分片（`optim_{step}_rank{N}.pt`）|
| 元数据 | rank 0 保存，含 dataloader 状态用于精确恢复 |

---

## 评估同步

`nanochat/loss_eval.py` 中各 rank 本地计算 loss，最后用 `dist.all_reduce(SUM)` 汇总为全局指标。

---

## 与标准 PyTorch DDP 的对比

| 特性 | 标准 DDP | nanochat DistMuonAdamW |
|------|---------|------------------------|
| 梯度同步时机 | backward 后自动 all_reduce | optimizer.step() 内异步手动控制 |
| 优化器状态 | 每 rank 完整副本 | ZeRO-2：大参数按 rank 分片 |
| 内存效率 | 较低 | 更高 |
| 控制粒度 | 低 | 高，可与自定义算法（Muon）深度集成 |

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `runs/speedrun.sh` | 训练启动脚本（torchrun）|
| `scripts/base_train.py` | 主训练循环 |
| `nanochat/common.py` | DDP 初始化工具函数 |
| `nanochat/dataloader.py` | 分布式数据加载与分片 |
| `nanochat/optim.py` | DistMuonAdamW 优化器 |
| `nanochat/checkpoint_manager.py` | 分布式 checkpoint 管理 |
| `nanochat/loss_eval.py` | 分布式评估 |
