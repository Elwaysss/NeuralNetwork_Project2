"""
上云第一步：检测昇腾 NPU 是否可用。
在 ModelArts 的 JupyterLab 终端里跑：python test_ascend_npu.py
看到最后的 "Ascend NPU 基本可用性测试完成" 就说明环境没问题。
"""
import os
import sys
import time

print("Python version:", sys.version)
print("PID:", os.getpid())
print("=" * 60)

try:
    import torch
    print("PyTorch version:", torch.__version__)
except Exception as e:
    print("PyTorch 导入失败:", repr(e))
    raise

print("=" * 60)

try:
    import torch_npu
    print("torch_npu version:", getattr(torch_npu, "__version__", "unknown"))
    HAS_TORCH_NPU = True
except Exception as e:
    print("torch_npu 导入失败:", repr(e))
    HAS_TORCH_NPU = False

print("=" * 60)

if not HAS_TORCH_NPU:
    print("当前环境没有可用的 torch_npu，无法继续测试 Ascend NPU。")
    print("可以先在终端检查：pip list | grep -E 'torch|npu'")
    sys.exit(1)

try:
    npu_available = torch.npu.is_available()
    npu_count = torch.npu.device_count()
    current_idx = torch.npu.current_device() if npu_count > 0 else None
    print("torch.npu.is_available():", npu_available)
    print("torch.npu.device_count():", npu_count)
    print("torch.npu.current_device():", current_idx)
except Exception as e:
    print("查询 NPU 状态失败:", repr(e))
    sys.exit(1)

print("=" * 60)

if not npu_available or npu_count == 0:
    print("没有检测到可用的 Ascend NPU。")
    sys.exit(1)

device = "npu:0"
try:
    torch.npu.set_device(device)
    print(f"已设置设备到 {device}")
except Exception as e:
    print("设置 NPU 设备失败:", repr(e))
    sys.exit(1)

print("=" * 60)

try:
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x @ y
    print("张量上 NPU 成功")
    print("x.device =", x.device, "z.device =", z.device)
except Exception as e:
    print("张量/矩阵乘法测试失败:", repr(e))
    sys.exit(1)

print("=" * 60)

try:
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inp = torch.randn(8, 16).to(device)
    target = torch.randn(8, 4).to(device)
    optimizer.zero_grad()
    loss = criterion(model(inp), target)
    loss.backward()
    optimizer.step()
    print("前向 + 反向传播成功, loss =", float(loss.detach().cpu()))
except Exception as e:
    print("训练测试失败:", repr(e))
    sys.exit(1)

print("=" * 60)

try:
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    for _ in range(3):
        c = a @ b
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(10):
        c = a @ b
    torch.npu.synchronize()
    print("性能 smoke test: 10 次 1024x1024 matmul 耗时 %.4f 秒" % (time.time() - t0))
except Exception as e:
    print("性能测试失败（前面功能可能已正常）:", repr(e))

print("=" * 60)
print("Ascend NPU 基本可用性测试完成。")
