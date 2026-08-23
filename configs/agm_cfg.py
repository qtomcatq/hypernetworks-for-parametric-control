from types import SimpleNamespace

cfg = SimpleNamespace(
    batch_size = 1024,
    scale = 5,
    scale_kin = 5,
    t_total = 1.0,
    t_size = 33,
    R_penalty = 0.0001,
    learning_rate = 0.005,
    iterations = 5000,
)    

