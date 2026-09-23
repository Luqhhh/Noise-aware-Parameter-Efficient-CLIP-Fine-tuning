"""GPU 健康探针。

B1（15:03）与 B2（16:00）都以同一个签名崩溃：
  trainer.py:1168 _gradient_norm -> optim.py:25
  torch.AcceleratorError: CUDA error: unknown error
两者的配置不同（anchor 0.5/lr 1e-5 vs anchor 0.0/lr 3e-6），唯一共有的是
机器与时间窗。在花 2.5 小时重跑之前，先确认这块卡现在能不能正常算。

刻意复现崩溃路径的两件事：
  1. 逐参数 grad 的 torch.stack([g.pow(2).sum()]).sum().sqrt() —— 就是崩掉的那个算子
  2. 每个迭代分配/释放约 200MB，制造 CachingHostAllocator 的压力
     （崩溃日志里出现过 "Exception in pinned allocator free()"）

跑满 90 秒。中途任何一次 CUDA 错误都会让脚本非零退出。
"""
import sys
import time

import torch

DEV = "cuda"
SECONDS = 90
CHURN_MB = 200


def main() -> int:
    if not torch.cuda.is_available():
        print("SMOKE FAIL: cuda.is_available() is False")
        return 2

    print(f"device: {torch.cuda.get_device_name(0)}")
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 750),
    ).to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    x = torch.randn(32, 512, device=DEV)

    churn_elems = CHURN_MB * 1024 * 1024 // 4  # float32
    t0 = time.time()
    iters = 0
    gnorm = float("nan")
    try:
        while time.time() - t0 < SECONDS:
            y = model(x)
            loss = y.pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # 崩溃现场的那个算子
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            gnorm = float(torch.stack([g.pow(2).sum() for g in grads]).sum().sqrt())
            opt.step()

            # 分配/释放压力，模拟训练时的 pinned allocator 抖动
            churn = torch.empty(churn_elems, device=DEV)
            churn.normal_()
            del churn

            iters += 1
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - 就是要抓住任何 CUDA 异常
        print(f"SMOKE FAIL after {iters} iters, {time.time() - t0:.1f}s: "
              f"{type(exc).__name__}: {exc}")
        return 1

    elapsed = time.time() - t0
    print(f"SMOKE OK iters={iters} last_grad_norm={gnorm:.4f} "
          f"elapsed={elapsed:.1f}s peak_mem={torch.cuda.max_memory_allocated()/1e6:.0f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
