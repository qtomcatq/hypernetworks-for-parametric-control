Hypernetwork-based Policies for Neural-ODE Control

This repository contains the official implementation for the paper "Hypernetwork-based policies for neural-ordinary-differential-equation control". We explore the intersection of Neural ODEs, Hypernetworks, and optimal control theory applied to linear systems and non-linear Kuramoto oscillator networks.

🚀 Overview
The project is divided into three primary control paradigms:

Neural Network Control (NNC): Policies generated via Hypernetworks for Neural ODE systems.

Optimal Control (OC): Classical benchmarks including LQR and Adjoint Gradient Methods (AGM).

Reinforcement Learning (PPO): Multi-Task Proximal Policy Optimization for synchronization tasks.

📂 Repository Structure:

🧠 Neural ODE-based Control (experiments/NNC/)
Implementations of Hypernetwork-based controllers.

Linear Systems: * Train_linear.ipynb: Policy optimization for canonical linear systems.

Test_linear.ipynb: Evaluation and benchmarking.

Kuramoto Networks:

Train_Kuramoto.ipynb: Learning synchronization policies.

Test_kuramoto.ipynb: Testing stability and order parameters.

⚖️ Classical & Adjoint Optimization (experiments/OC/)
Baseline control methods and gradient-based optimization.

LQR: OC/Linear/LQR.ipynb – Finite-horizon LQR using batched Riccati solvers.

AGM / v-AGM: OC/AGM/AGM.ipynb and v-AGM.ipynb – Adjoint Gradient Methods for direct control signal optimization (standard and vectorized versions).

🤖 Reinforcement Learning (experiments/ppo/)
MT-PPO: ppo/ppo.ipynb – Multi-Trajectory PPO implementation for robust policy learning in stochastic environments.

🛠 Installation & Setup
Clone the repository:

Install Dependencies:
This project requires JAX for fast environment simulation and PyTorch for neural network modeling.

Core Modules:
The core/ folder contains shared utilities. 
