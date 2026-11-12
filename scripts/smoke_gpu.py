"""GPU smoke test: exits 0 only if this machine can train on its CUDA device.

Run with:  uv run python scripts/smoke_gpu.py

The Quadro P4000 is a Pascal card (compute capability 6.1). PyTorch wheels built for
CUDA 12.8 and later no longer include kernels for it, so this script checks that the
installed build still lists the device's architecture and can run a real training step.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed. Run: uv sync --extra train")
        return 1

    print(f"torch          {torch.__version__}")
    print(f"cuda build     {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return 1

    device = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    props = torch.cuda.get_device_properties(device)
    print(f"device         {props.name}")
    print(f"vram           {props.total_memory / 2**30:.1f} GiB")
    print(f"device arch    {arch}")
    print(f"build archs    {' '.join(arch_list)}")
    if arch not in arch_list:
        print(f"FAIL: this torch build has no kernels for {arch}")
        return 1

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(768, 768), torch.nn.GELU(), torch.nn.Linear(768, 2)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(64, 768, device=device)
    targets = torch.randint(0, 2, (64,), device=device)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    torch.cuda.synchronize()

    print(f"loss           {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"peak memory    {torch.cuda.max_memory_allocated(device) / 2**20:.0f} MiB")
    if not losses[-1] < losses[0]:
        print("FAIL: loss did not decrease over 5 steps")
        return 1

    print("OK: GPU training works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
