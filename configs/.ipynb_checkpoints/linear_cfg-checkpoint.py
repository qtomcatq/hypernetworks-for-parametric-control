from types import SimpleNamespace
import numpy as np

cfg = SimpleNamespace(
    dim=128,
    batch_size=1024,
    weight=0.1,    # trade-off weight
    learning_rate=0.001,
    alpha=0.999,    # EMA coefficient
    gamma=0.9999,   # Scheduler decay
    iterations=10000,
    clip_val=1e2,   # weight clip
    t_total=1.0,    # time horizon
    t_size=33,
    strategy="ENC", #choose strategy: HNC or ENC
    activation="SiLU", 
    std_scales=(np.sqrt(12), np.sqrt(12)), #normalization for hypernetwork
    score= 1e6,
    skip=True,
)


# Global Test Config
eval_cfg = SimpleNamespace(
    dim = 128,
    batch_size = 1024,
    weight = 0.1,
    clip = 100,
    t_total = 1.0,
    t_size = 33,
    test_iters = 2500,
    strategy = "ENC",
    skip = True,
    activation = "SiLU",
    std_scales = (np.sqrt(12), np.sqrt(12)),
)
