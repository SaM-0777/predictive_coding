# Predictive Coding: Supervised Learning & Reinforcement Learning

A PyTorch implementation of **Predictive Coding (PC)** exploring biologically plausible credit assignment across supervised image classification and deep reinforcement learning.

This project implements the theoretical framework established by [Millidge et al. (2020) — Predictive Coding Approximates Backprop along Arbitrary Computation Graphs](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F2006.04182), demonstrating how local, Hebbian-style learning rules operating via iterative energy minimization can approximate backpropagation. The repository includes:

1. **Supervised Learning (`src/pc_model.py`)**: A multi-layer predictive coding network evaluated on MNIST, through iterative activation relaxation and local weight updates.
2. **Reinforcement Learning (`src/pc_agent.py`)**: An experimental Proximal Policy Optimization (PPO) agent integrating predictive coding principles to train an actor-critic model on interactive environments like `Catcher-PLE-v0`.


## Repository setup

The project requires Python 3.12 or newer. Its Python dependencies are declared in [`pyproject.toml`](pyproject.toml), and [`uv.lock`](uv.lock) records the locked environment.

```bash
git clone https://github.com/SaM-0777/predictive_coding.git
cd predictive_coding

uv sync
source .venv/bin/activate

# Required by torch.utils.tensorboard.SummaryWriter in pc_agent.py.
uv add tensorboard
```

The core environment includes PyTorch, torchvision, Gymnasium, NumPy, Tyro, and the other packages listed in `pyproject.toml`. The agent writes metrics through `torch.utils.tensorboard.SummaryWriter`, so the setup also installs the `tensorboard` package.

## Train `pc_model` on MNIST

`src/pc_model.py` defines a convolutional predictive-coding model with the following stages:

1. Convolution and ReLU
2. Max pooling
3. Convolution and ReLU
4. Flattening
5. Three fully connected prediction layers

Each training batch initializes the activities, runs the configured number of inference steps, evaluates the final predictions, and updates the prediction-function parameters using local error signals. The default training configuration uses 10 epochs, batch size 64, 20 inference steps, and a parameter learning rate of `5e-4`.

Run the predictive-coding MNIST trainer from the repository root:

```bash
uv run python -m src.pc_model
```


## Train `pc_agent` with PPO on Catcher

`src/pc_agent.py` trains a PPO agent on Catcher using a predictive-coding three-phase update:

1. Collect vectorized rollouts and compute PPO advantages and returns with GAE.
2. Convert the PPO objective into actor and critic task signals, then perform iterative activity inference through the shared convolutional trunk and the actor and critic heads.
3. Apply local predictive-coding parameter updates, using a shared PyTorch optimizer or direct local updates.

The default agent configuration uses eight synchronized environments, 128 steps per rollout, 500,000 total timesteps, four PPO update epochs, and Adam for the parameter updates. Observations are rendered from the Pygame Learning Environment, converted to 84×84 pixel observations, converted to grayscale, and stacked over four frames. Training metrics are written to `runs/<run-name>` for TensorBoard.

### Install `gym-games`

The Catcher environment comes from the [`qlan3/gym-games`](https://github.com/qlan3/gym-games) repository. Install it inside the project environment:

```bash
uv pip install "git+https://github.com/qlan3/gym-games.git"
```

The `gym-games` package supplies the Gymnasium-compatible Pygame environments, including `Catcher-PLE-v0`, and declares its Pygame Learning Environment dependencies. Its setup instructions also specify platform prerequisites for Pygame.

### Environment Dependency: `gym-games` Setup
Training on the `Catcher-PLE-v0` environment requires the [PyGame Learning Environment (PLE)](https://github.com/ntasfi/PyGame-Learning-Environment) and [gym-games](https://github.com/qlan3/gym-games). Follow these installation steps:

#### Step 1: Install System Dependencies for Pygame

* **Ubuntu / Debian Linux:**
  ```bash
  sudo apt-get update
  sudo apt-get install -y python3-pygame libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
  pip install pygame==2.5.2
  ```

* **macOS (via Homebrew):**
```bash
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf pkg-config
pip install pygame==2.5.2
```

#### Step 2: Install PyGame Learning Environment (PLE)
```bash
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment
pip install -e .
cd ..
```

#### Step 3: Install MinAtar & `gym-games`
```bash
pip install minatar==1.0.15
pip install git+https://github.com/qlan3/gym-games.git
```

### Run the Catcher trainer

Run the default PPO and predictive-coding configuration:

```bash
uv run python -m src.pc_agent
```

Useful configuration examples use the Tyro command-line interface exposed by `pc_agent.py`:

```bash
# Use direct local predictive-coding parameter updates.
uv run python -m src.pc_agent --optimizer none

# Enable video capture for environment 0.
uv run python -m src.pc_agent --capture-video
```

The supported optimizer values are `Adam`, `SGD`, `RMSprop`, and `none`. The trainer prints PPO objective diagnostics, approximate KL divergence, explained variance, and steps per second. Add `--save-model` to save the agent state and training configuration under `runs/<run-name>/`.

View logged metrics with TensorBoard:

```bash
uv run tensorboard --logdir runs
```

## Conclusion

The repository provides a single PyTorch codebase for studying predictive coding across two settings: MNIST classification with local layer updates and Catcher reinforcement learning with PPO-derived task signals. The MNIST path is available through `src.pc_model`, while `src.pc_agent` provides the Catcher training loop after installing the `gym-games` environment package and its Pygame prerequisites.

---

## References

* Millidge, B., Tschantz, A., & Buckley, C. L. (2020). *Predictive Coding Approximates Backprop along Arbitrary Computation Graphs*. [arXiv:2006.04182](https://arxiv.org/abs/2006.04182).
* Rao, R. P., & Ballard, D. H. (1999). *Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects*. Nature Neuroscience, 2(1), 79-87.
* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).
* Lan, Q. (2021). *gym-games: Collection of Reinforcement Learning environments*. [GitHub Repository](https://github.com/qlan3/gym-games).
