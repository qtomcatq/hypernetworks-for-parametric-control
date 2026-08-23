from types import SimpleNamespace

cfg = SimpleNamespace(
    weight = 0.1,
    N = 128,
    iter_count = 1000,
    nsteps = 32,
    batch_size = 2 * 1024,
    total_time = 1.0,
)