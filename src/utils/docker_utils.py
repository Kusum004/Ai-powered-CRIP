# src/utils/docker_utils.py
"""Docker environment detection, health check, and system diagnostics."""
import os
import platform
import psutil

def is_running_in_docker() -> bool:
    """Detects if the application is running inside a Docker container."""
    return os.path.exists('/.dockerenv') or os.getenv('IS_DOCKER', 'false').lower() == 'true'

def get_system_health() -> dict:
    """Returns runtime system performance metrics for monitoring."""
    cpu_percent = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))
    
    return {
        "is_docker": is_running_in_docker(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_usage_pct": cpu_percent,
        "memory_used_mb": round(memory.used / (1024 * 1024), 2),
        "memory_total_mb": round(memory.total / (1024 * 1024), 2),
        "memory_pct": memory.percent,
        "disk_free_gb": round(disk.free / (1024 * 1024 * 1024), 2)
    }
