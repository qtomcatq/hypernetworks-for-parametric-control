from types import SimpleNamespace
#configuration for training
k_cfg = SimpleNamespace(
    graph="erdos",
    connectivity=0.3,
    state_size=64,
    batch_size=256,
    weight=1e-4,
    scale=5,
    scale_kin=5,
    lr=1e-4,
    gamma=0.999,
    alpha=0.995,
    iterations=5000,
    t_total=1.0,
    t_size=33,
    strategy="ENC", #select strategy: ENC, HNC, HNCt
    activation="ReLU",
    clip=100.0,
    skip=False,
)

#configuration for testing
eval_cfg = SimpleNamespace(
    state_size=64,
    batch_size=1024,
    weight=1e-4,
    scale=5,
    scale_kin=5,
    t_size=33,
    t_total=1.0,
    iter=100,
    strategy="ENC",
    activation="ReLU",
    skip=False
)

