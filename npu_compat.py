"""
环境兼容补丁。

华为云这套镜像里 triton 的版本和 torch 2.6 的 inductor 对不上，
import torch 时会自动加载 torch_npu，进而导入 inductor，报
"cannot import name 'AttrsDescriptor' from 'triton.compiler.compiler'"。
我们在 NPU 上是 eager 执行，根本用不到 triton/inductor，所以在 import torch
之前给 triton 补一个占位的 AttrsDescriptor，让导入链能走通。

注意：必须在 import torch 之前先 import 本模块。
"""
import dataclasses
import importlib


def _patch():
    try:
        import triton  # noqa: F401
    except Exception:
        return

    @dataclasses.dataclass
    class _AttrsDescriptor:
        pass

    for name in ("triton.backends.compiler", "triton.compiler.compiler"):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if not hasattr(mod, "AttrsDescriptor"):
            mod.AttrsDescriptor = _AttrsDescriptor


_patch()
