"""The device a kernel runs on: its arch name and arch family (the backend priority's keys)."""

import torch


def arch_of(device) -> str:
    """``smXY`` of a CUDA device, else the device type."""
    device = torch.device(device)
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return f"sm{major}{minor}"
    return device.type


# Arch families sharing one set of kernels: sm10x the datacenter Blackwell cards (B200 sm100,
# B300 sm103; 228 KiB of shared memory per SM, tcgen05), sm12x the consumer ones (RTX 50xx
# sm120, DGX Spark sm121; about 100 KiB)
FAMILIES = {"sm10x": ("sm100", "sm101", "sm103"), "sm12x": ("sm120", "sm121")}


def family_of(arch: str) -> str:
    """The family an arch's kernels come from (``FAMILIES``), else the arch itself."""
    return next((f for f, members in FAMILIES.items() if arch in members), arch)


def family(device) -> str:
    """The kernel family of a device."""
    return family_of(arch_of(device))
