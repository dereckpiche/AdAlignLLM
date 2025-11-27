"""
File: mllm/utils/resource_context.py
Summary: Tracks system resource usage via a context manager.
"""

import logging
import time
from contextlib import contextmanager

import torch


def vram_usage():
    output = ""
    for i in range(torch.cuda.device_count()):
        gpu_memory_allocated = torch.cuda.memory_allocated(i) / (
            1024**3
        )  # Convert bytes to GB
        gpu_memory_reserved = torch.cuda.memory_reserved(i) / (
            1024**3
        )  # Convert bytes to GB
        output += f"GPU {i}: Memory Allocated: {gpu_memory_allocated:.2f} GB, Memory Reserved: {gpu_memory_reserved:.2f} GB"
    return output


def ram_usage():
    import psutil

    process = psutil.Process()
    memory_info = process.memory_info()
    ram_used = memory_info.rss / (1024**3)  # Convert bytes to GB
    return f"RAM Usage: {ram_used:.2f} GB"


@contextmanager
def resource_logger_context(logger: logging.Logger, task_description: str):
    """
    Context manager to log the resource usage of the current task.
    Args:
        logger: The logger to use to log the resource usage.
        task_description: The description of the task to log.
    Returns:
        None
    """
    try:
        initial_time = time.time()
        # Assume CUDA is available and use device 0 only
        total_mem_bytes = torch.cuda.get_device_properties(0).total_memory
        initial_total_bytes = torch.cuda.memory_allocated(
            0
        ) + torch.cuda.memory_reserved(0)
        torch.cuda.reset_peak_memory_stats(0)
        yield None
    finally:
        final_time = time.time()
        # Ensure kernels within the block are accounted for
        torch.cuda.synchronize()

        # Compute metrics
        final_allocated_bytes = torch.cuda.memory_allocated(0)
        final_reserved_bytes = torch.cuda.memory_reserved(0)
        final_total_bytes = final_allocated_bytes + final_reserved_bytes

        delta_vram_percent_total = (
            100 * (final_total_bytes - initial_total_bytes) / total_mem_bytes
            if total_mem_bytes
            else 0.0
        )
        current_percent_vram_taken = (
            100 * final_total_bytes / total_mem_bytes if total_mem_bytes else 0.0
        )
        block_peak_percent = (
            100 * torch.cuda.max_memory_allocated(0) / total_mem_bytes
            if total_mem_bytes
            else 0.0
        )
        delta_time_str = time.strftime(
            "%H:%M:%S", time.gmtime(final_time - initial_time)
        )

        logger.info(
            f"For task: {task_description}, ΔVRAM % (total): {delta_vram_percent_total:.2f}%, Current % of VRAM taken: {current_percent_vram_taken:.2f}%, Block Peak % of device VRAM: {block_peak_percent:.2f}%, ΔTime: {delta_time_str}"
        )
