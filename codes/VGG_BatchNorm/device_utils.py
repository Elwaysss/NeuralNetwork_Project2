"""设备选择：昇腾 NPU 优先，没有就退回 CPU（本地调试用）。"""
import npu_compat  # noqa: F401  必须在 import torch 之前
import torch


def get_device():
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return torch.device("npu:0")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
