from types import SimpleNamespace
# Physics and Training Hyperparameters

cfg = SimpleNamespace(
    # Environment & Physics
    total_timesteps = 100000,
    steps_per_trajectory = 32,
    batch_size = 1024,
    gamma = 0.999999,
    gae_lambda = 0.999999,
    
    # Training
    epochs_per_config = 10000,
    learning_rate = 1e-3,
    clip_epsilon = 0.2,
    noise_rms = 32.0,
    alpha_ema = 0.995,
    reward_weight_r= 1e-4,
    
    # Architecture
    omniscent=True
)
