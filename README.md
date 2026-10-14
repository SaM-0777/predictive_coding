# Predictive Coding

This repository contains PyTorch implementations of predictive-coding learning for supervised image classification and reinforcement learning. The MNIST model in [`src/pc_model.py`](src/pc_model.py) performs iterative activity inference through a convolutional network and applies local prediction-error updates to its parameters. The Catcher agent in [`src/pc_agent.py`](src/pc_agent.py) combines PPO rollout collection with predictive-coding inference and local parameter updates for the `PLE/Catcher` game environment.

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

On macOS, install the SDL libraries and Pygame package listed by `gym-games`:

```bash
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf pkg-config
uv pip install pygame==2.5.2
```

On Ubuntu, install the system Pygame package and the matching Python package:

```bash
sudo apt-get -y install python-pygame
uv pip install pygame==2.5.2
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

