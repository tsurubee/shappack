import math
from pathlib import Path


def get_cpu_quota_within_docker():
    """Check the cgroup configuration for CPU limits in order to handle
    the case of running in a Docker container.
    """
    cpu_cores = None
    cfs_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    cfs_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")

    if cfs_period.exists() and cfs_quota.exists():
        with cfs_period.open("rb") as p, cfs_quota.open("rb") as q:
            period, quota = int(p.read()), int(q.read())
            cpu_cores = math.ceil(quota / period) if quota > 0 and period > 0 else None

    return cpu_cores
