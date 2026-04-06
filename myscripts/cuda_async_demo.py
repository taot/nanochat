"""
验证 CUDA 操作对 CPU 是异步的。

方案1：timing 对比
  - 不加 synchronize：GPU 做大量矩阵乘法，CPU 几乎立刻返回
  - 加 torch.cuda.synchronize()：CPU 等 GPU 真正算完

方案2：直接观察
  - 启动 GPU 计算后，CPU 立刻在 GPU 算完之前就能继续执行
"""
import time
import torch
import datetime

assert torch.cuda.is_available(), "需要 CUDA GPU"

N = 8192  # 足够大，让 GPU 跑一段时间
a = torch.randn(N, N, device="cuda")
b = torch.randn(N, N, device="cuda")

# 先 warmup，避免第一次 kernel launch 的额外开销
_ = a @ b
torch.cuda.synchronize()

print("=" * 60)
print("方案1：timing 对比")
print("=" * 60)

def print_t(s: str) -> None:
    print(f"{s}: {datetime.datetime.now()}: {s}")

# ── 不 synchronize ──────────────────────────────────────────────
print_t("开始")
t0 = time.perf_counter()
c = a @ b          # 只是把 kernel 推入 CUDA stream，CPU 立刻返回
t1 = time.perf_counter()
cpu_return_time = (t1 - t0) * 1000
print(f"[不 sync] CPU 从 a@b 返回耗时: {cpu_return_time:.3f} ms  ← GPU 还没算完！")

print_t("after cpu_return_time")

# ── 加 synchronize ───────────────────────────────────────────────
t0 = time.perf_counter()
print_t("before c = a @ b")
c = a @ b
print_t("after c = a @ b")

d = a @ b
print_t("after d = a @ b")

torch.cuda.synchronize()   # 阻塞 CPU 直到 GPU 完成

print_t("before time.perf_counter()")
t1 = time.perf_counter()
print_t("after time.perf_counter()")

gpu_total_time = (t1 - t0) * 1000
print(f"[加 sync] GPU 真正算完耗时:    {gpu_total_time:.3f} ms")
print(f"GPU 实际计算时间约是 CPU 返回时间的 {gpu_total_time / cpu_return_time:.0f}x")


print()
print("=" * 60)
print("方案2：直接观察 CPU 在 GPU 算完前就能继续执行")
print("=" * 60)

# 启动 GPU 计算（不 sync）
c = a @ b
cpu_time = time.perf_counter()
print(f"CPU 时间戳（a@b 返回后立刻）: t = 0 ms")

# CPU 继续执行其他工作（这里模拟 dataloader 的 CPU 准备工作）
time.sleep(0.001)  # 1ms CPU 工作
print(f"CPU 做了 1ms 的工作...")

# 现在才等 GPU
torch.cuda.synchronize()
gpu_done_time = time.perf_counter()
print(f"GPU 完成时间: t = {(gpu_done_time - cpu_time) * 1000:.3f} ms")
print()
print("=> GPU 算完的时刻比 CPU 开始做其他工作晚，说明 CPU 确实先于 GPU 返回")
