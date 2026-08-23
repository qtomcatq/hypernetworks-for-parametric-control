from types import SimpleNamespace

cfg = SimpleNamespace(
    batch_size = 1024,
    scale = 5,
    scale_kin = 5,
    t_total = 1.0,
    t_size = 33,
    R_penalty = 0.0001,
    lr_base = 0.005,
    iterations = 6000,
    # Decay schedules (sched1 and sched2 from your original code)
    lr_decay = 1.0, 
)