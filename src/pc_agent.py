from dataclasses import dataclass
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterable, cast

import gymnasium as gym
from gymnasium import spaces
from gym_pygame.envs.catcher import CatcherEnv
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

Activation = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class Args:
    """the name of this experiment"""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """seed of the experiment"""
    seed: int = 1
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    torch_deterministic: bool = True
    """device"""
    device: str = "cuda" if torch.cuda.is_available() else "mps"
    """if toggled, this experiment will be tracked with Weights and Biases"""
    track: bool = False
    """the wandb's project name"""
    wandb_project_name: str = "cleanRL"
    """the entity (team) of wandb's project"""
    wandb_entity: str | None = None
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    capture_video: bool = False

    """the id of the environment"""
    env_id: str = "PLE/Catcher"
    """total timesteps of the experiments"""
    total_timesteps: int = 500_000
    """the learning rate of the optimizer"""
    learning_rate: float = 2.5e-4
    """the number of parallel game environments"""
    num_envs: int = 8
    """the number of steps to run in each environment per policy rollout"""
    num_steps: int = 128
    """Toggle learning rate annealing for policy and value networks"""
    anneal_lr: bool = True
    """the discount factor gamma"""
    gamma: float = 0.99
    """the lambda for the general advantage estimation"""
    gae_lambda: float = 0.95
    """the number of mini-batches"""
    num_minibatches: int = 4
    """the K epochs to update the policy"""
    update_epochs: int = 4
    """Toggles advantages normalization"""
    norm_adv: bool = True
    """the surrogate clipping coefficient"""
    clip_coef: float = 0.1
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    clip_vloss: bool = True
    """coefficient of the entropy"""
    ent_coef: float = 0.01
    """coefficient of the value function"""
    vf_coef: float = 0.5
    """the maximum norm for the gradient clipping"""
    max_grad_norm: float = 0.5
    """the target KL divergence threshold"""
    target_kl: float | None = None

    # Predictive-coding settings.
    pc_inference_steps: int = 8
    pc_inference_lr: float = 0.1
    pc_error_clip: float = 50.0
    pc_parameter_lr: float = 1e-4
    # None uses the direct local parameter update.
    # Otherwise choose a torch.optim optimizer, e.g. "Adam" or "SGD".
    optimizer: str | None = None

    """the batch size (computed in runtime)"""
    batch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the number of iterations (computed in runtime)"""
    num_iterations: int = 0
    """save model"""
    save_model: bool = False


class PLEPixelObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        inner = cast(CatcherEnv, env.unwrapped)
        h = int(inner.game.height)
        w = int(inner.game.width)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, 3), dtype=np.uint8
        )

    def observation(self, observation: Any) -> np.ndarray:
        inner = cast(CatcherEnv, self.env.unwrapped)
        rgb = inner.gameOb.getScreenRGB()  # (W, H, 3)
        return np.transpose(rgb, (1, 0, 2)).copy()


def make_env(idx, capture_video, run_name, display=False):
    def thunk():
        env = CatcherEnv(normalize=False, display=display, width=84, height=84)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = PLEPixelObs(env)
        # env = NoopResetEnv(env, noop_max=30)
        # env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore
        #    env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env

    return thunk


class PCLayer(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        activation_fn: Activation = lambda x: x,
        lr: float = 1e-4,
        gradient_clip: float | None = 50.0,
    ) -> None:
        super().__init__()

        self.layer = layer
        self.activation_fn = activation_fn
        self.lr = lr
        self.gradient_clip = gradient_clip
        self.input: torch.Tensor | None = None

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(self.layer(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward prediction of a generic differentiable module.

        Mathematical formulation:
            v_hat_j = f_j(v_i; theta_j)

        where:
            v_i       : input activity of the current module.
            f_j       : the module's differentiable prediction function.
            theta_j   : the module's parameters.
            v_hat_j   : predicted activity at the child vertex.

        The input activity is stored so that the local prediction function
        can be reconstructed when calculating its derivative during backward().

        Returns:
            v_hat_j: the module's predicted output.
        """
        self.input = x.detach().clone()
        return self._predict(x)

    def backward(self, prediction_error: torch.Tensor) -> torch.Tensor:
        """
        Calculate the derivative-weighted error contribution from a child
        prediction to its parent activity, using PyTorch autograd.

        Mathematical formulation:
            v_hat_j = f_j(v_i; theta_j)
            J_ji = d(v_hat_j) / d(v_i)
            dx = J_ji^T @ e

        For predictive coding, when e = epsilon_j, where epsilon_j is the
        child's prediction error, this calculates:

            dx = (d(v_hat_j) / d(v_i))^T @ epsilon_j

        This is the derivative-weighted child-error contribution appearing
        in Equation 2 of Millidge et al. (2020).

        The complete predictive-coding activity dynamics for parent vertex i,
        using the equation's notation, are:

            dv_i/dt = epsilon_i - sum_{j in C(v_i)} J_ji^T @ epsilon_j

        where:
            epsilon_i : local prediction error at the parent vertex.
            C(v_i)    : set of child vertices of the parent.
            J_ji      : Jacobian of child j's prediction with respect to
                        parent i's activity.

        This method calculates one child contribution. The caller must sum
        contributions from all children and combine them with the parent's
        local prediction error to implement the full activity equation.

        PyTorch autograd computes the vector-Jacobian product without
        explicitly constructing the Jacobian matrix.

        Args:
            prediction_error: Incoming error signal. For the predictive-coding interpretation,
                this must be the corresponding child's prediction error.

        Returns:
            dx: Derivative-weighted error contribution with respect to the
                module's input activity.
        """
        if self.input is None:
            raise RuntimeError("Call forward() before backward().")

        with torch.enable_grad():
            x = self.input.detach().requires_grad_(True)
            prediction = self._predict(x)
            (dx,) = torch.autograd.grad(
                outputs=prediction,
                inputs=x,
                grad_outputs=prediction_error.detach(),
                create_graph=False,
                retain_graph=False,
            )

        return dx.detach()

    def update_weights(
        self,
        prediction_error: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        """
        Update the prediction-function parameters according to Equation 3
        of Millidge et al. (2020), Predictive Coding Approximates Backprop
        Along Arbitrary Computation Graphs.

        Paper's equations:

            v_hat_i = f_i(x; theta_i)
            epsilon_i = v_i - v_hat_i
            d(theta_i)/dt = -dF/d(theta_i)
                          = epsilon_i^T * d(v_hat_i)/d(theta_i)

        Discrete parameter update:

            theta_i <- theta_i + learning_rate * [d(v_hat_i)/d(theta_i)]^T * epsilon_i

        This method calculates the local parameter-update direction using
        PyTorch autograd and applies it after activity inference converges.

        Args:
            prediction_error:
                The local prediction error epsilon_i = v_i - v_hat_i
                associated with this prediction function's output.
        """
        # The prediction function requires its feedforward input x:
        # v_hat_i = f_i(x; theta_i).
        # The cached input is used to evaluate the local parameter derivative.
        if self.input is None:
            raise RuntimeError("Call forward() before update_weights().")

        # theta_i denotes the trainable parameters of the prediction function.
        # Equation 3 requires derivatives with respect to these parameters:
        # d(theta_i)/dt = -dF/d(theta_i).
        parameters = tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )

        if not parameters:
            return None

        # Treat the prediction-function input as fixed while differentiating
        # with respect to theta_i:
        # v_hat_i = f_i(x; theta_i).
        x = self.input.detach()

        with torch.enable_grad():
            # v_hat_i = activation_fn(layer(x; theta_i)).
            prediction = self._predict(x)

            # Calculate the local parameter-update direction:
            #
            # g_theta_i = [d(v_hat_i)/d(theta_i)]^T * epsilon_i.
            #
            # This is the derivative term in Equation 3.
            # PyTorch calculates the vector-Jacobian product directly.
            # prediction_error is held fixed during this derivative.
            dW = torch.autograd.grad(
                outputs=prediction,
                inputs=parameters,
                grad_outputs=prediction_error.detach(),
                allow_unused=True,
                create_graph=False,
                retain_graph=False,
            )

        # Apply the positive parameter-update direction specified by Equation 3:
        #
        # d(theta_i)/dt = epsilon_i^T * d(v_hat_i)/d(theta_i).
        #
        # torch.no_grad() ensures that the in-place parameter update does not
        # become part of another autograd computation graph.
        with torch.no_grad():
            for parameter, update in zip(parameters, dW):
                if update is None:
                    continue
                # A parameter independent of this prediction has no local
                # derivative contribution: d(v_hat_i)/d(parameter) = 0.
                # update = 2.0 * update
                if self.gradient_clip is not None:
                    update = update.clamp(
                        min=-self.gradient_clip, max=self.gradient_clip
                    )

                if optimizer is None:
                    # Forward Euler parameter update:
                    #
                    # theta_i^(t+1) = theta_i^t + eta_theta * g_theta_i.
                    parameter.add_(update, alpha=self.lr)
                else:
                    # PyTorch optimizers minimize their objective, so pass the
                    # negative of the paper's positive update direction.
                    parameter.grad = (-update).detach().clone()

        return dW


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        T: int = 5,
        inference_lr: float = 0.01,
        parameter_lr: float = 1e-4,
    ) -> None:
        super().__init__()

        self.T = T
        self.inference_lr = inference_lr
        self.parameter_lr = parameter_lr

        action_dim = envs.single_action_space.n
        self.network = nn.Sequential(
            PCLayer(
                nn.Conv2d(4, 32, 8, stride=4),
                activation_fn=torch.relu,
                lr=parameter_lr,
            ),
            PCLayer(
                nn.Conv2d(32, 64, 4, stride=2),
                activation_fn=torch.relu,
                lr=parameter_lr,
            ),
            PCLayer(
                nn.Conv2d(64, 64, 3, stride=1),
                activation_fn=torch.relu,
                lr=parameter_lr,
            ),
            PCLayer(nn.Flatten(), lr=parameter_lr),
            PCLayer(
                nn.Linear(64 * 7 * 7, 512), activation_fn=torch.relu, lr=parameter_lr
            ),
        )
        self.critic = PCLayer(nn.Linear(512, 1))
        self.actor = PCLayer(nn.Linear(512, action_dim))

        self.L = len(self.network)
        # predictions
        self.network_predictions: list[torch.Tensor] = [
            torch.empty(0) for _ in range(self.L + 1)
        ]
        self.critic_prediction: list[torch.Tensor] = [torch.empty(0)]
        self.actor_prediction: list[torch.Tensor] = [torch.empty(0)]
        # activations
        self.network_activations: list[torch.Tensor] = [
            torch.empty(0) for _ in range(self.L + 1)
        ]
        self.critic_activation: list[torch.Tensor] = [torch.empty(0)]
        self.actor_activation: list[torch.Tensor] = [torch.empty(0)]
        # prediction error
        self.network_prediction_errors: list[torch.Tensor] = [
            torch.empty(0) for _ in range(self.L + 1)
        ]
        self.critic_prediction_error: list[torch.Tensor] = [torch.empty(0)]
        self.actor_prediction_error: list[torch.Tensor] = [torch.empty(0)]
        # inference error signals
        self.network_inference_error_signals: list[torch.Tensor] = [
            torch.empty(0) for _ in range(self.L + 1)
        ]
        self.critic_inference_error_signal: list[torch.Tensor] = [torch.empty(0)]
        self.actor_inference_error_signal: list[torch.Tensor] = [torch.empty(0)]

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None, return_outputs=False):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(hidden)

        if return_outputs:
            return action, logprob, entropy, value, logits

        return action, logprob, entropy, value, None

    def initialize(self, obs: torch.Tensor):
        with torch.no_grad():
            x = obs / 255.0

            self.network_predictions[0] = x
            self.network_activations[0] = x

            self.network_prediction_errors[0] = torch.zeros_like(x)
            self.network_inference_error_signals[0] = torch.zeros_like(x)

            for i, l in enumerate(self.network):
                prediction = l(self.network_predictions[i])

                self.network_predictions[i + 1] = prediction.detach().clone()
                self.network_activations[i + 1] = prediction.detach().clone()

                # No mismatch exists at initialization because activity
                # equals prediction.
                self.network_prediction_errors[i + 1] = torch.zeros_like(prediction)
                self.network_inference_error_signals[i + 1] = torch.zeros_like(
                    prediction
                )

            # Actor branch
            hidden_activity = self.network_activations[self.L]
            actor_prediction = self.actor(hidden_activity)
            self.actor_prediction[0] = actor_prediction
            self.actor_activation[0] = actor_prediction
            self.actor_prediction_error[0] = torch.zeros_like(actor_prediction)
            self.actor_inference_error_signal[0] = torch.zeros_like(actor_prediction)

            # Critic Branch
            critic_prediction = self.critic(hidden_activity)
            self.critic_prediction[0] = critic_prediction
            self.critic_activation[0] = critic_prediction
            self.critic_prediction_error[0] = torch.zeros_like(critic_prediction)
            self.critic_inference_error_signal[0] = torch.zeros_like(critic_prediction)

            # PPO task signals
            self.actor_task_signal = torch.zeros_like(actor_prediction)
            self.critic_task_signal = torch.zeros_like(critic_prediction)

        return actor_prediction, critic_prediction

    def initialize_ppo_signal(self, ppo_loss, actor_logits, critic_value):
        """
        Derive actor and critic task signals from the existing PPO objective.

        The task signals are the negative derivatives of the PPO loss
        with respect to the actor logits and critic value:

            actor_task_signal = -d(L_PPO) / d(actor_logits)
            critic_task_signal = -d(L_PPO) / d(critic_value)

            s_a = -d(L_PPO) / dz
            s_c = -d(L_PPO) / dv

            where:
            - z represents the actor's logits.
            - v represents the critic's predicted values.
            - s_a is the actor task signal.
            - s_c is the critic task signal.

        These are external task signals, not local prediction errors.

        Args:
            ppo_loss: Scalar PPO objective for the current minibatch.
            actor_logits: Exact actor-logit tensor used to calculate ppo_loss.
            critic_value: Exact critic-value tensor used to calculate ppo_loss.
        """
        with torch.enable_grad():
            actor_gradient, critic_gradient = torch.autograd.grad(
                outputs=ppo_loss,
                inputs=(actor_logits, critic_value),
                retain_graph=False,
                create_graph=False,
            )

            # Negative loss gradients provide the task-directed signals.
            self.actor_task_signal = -actor_gradient.detach()
            self.critic_task_signal = -critic_gradient.detach()

        return self.actor_task_signal, self.critic_task_signal

    def inference(self, error_clip: float | None = None):
        """
        Perform predictive-coding activity inference.

        Local prediction errors:
            epsilon_i = mu_i - mu_hat_i

        For hidden activities, the gradient of the combined energy is:
            dE/dmu_i = epsilon_i - sum_j J_ji^T @ epsilon_j
                    - task contribution from downstream branches

        For the actor and critic outputs:
            dE/dz = epsilon_actor - actor_task_signal
            dE/dv = epsilon_critic - critic_task_signal

        Activities are updated by gradient descent:
            mu_i <- mu_i - inference_lr * dE/dmu_i

        Args:
            inference_steps: Number of activity-inference iterations.
            inference_lr: Step size for activity updates.
            error_clip: Optional clipping threshold for local prediction
                errors. If None, errors are not clipped.

        Returns:
            None. Updates the stored activities, predictions, local
            prediction errors, and inference error signals in place.
        """
        if self.T <= 0:
            raise ValueError("inference_steps must be greater then 0")
        if self.inference_lr < 0:
            raise ValueError("inference_lr must be non-negative")
        if self.network_activations[0].numel() == 0:
            raise RuntimeError(
                "Call initialize(observation) before infer_activities()."
            )
        if self.actor_task_signal.numel() == 0 or self.critic_task_signal.numel() == 0:
            raise RuntimeError(
                "Task signals are not initialized. Initialize the agent "
                "and establish PPO task signals before inference."
            )

        def clip_error(error: torch.Tensor) -> torch.Tensor:
            if error_clip is None:
                return error

            return error.clamp(min=-error_clip, max=error_clip)

        with torch.no_grad():
            for _ in range(self.T):

                for i, layer in enumerate(self.network):
                    prediction = layer(self.network_activations[i])
                    self.network_predictions[i + 1] = prediction.detach()

                hidden_activity = self.network_activations[self.L]

                actor_prediction = self.actor(hidden_activity)
                critic_prediction = self.critic(hidden_activity)

                self.actor_prediction[0] = actor_prediction.detach()
                self.critic_prediction[0] = critic_prediction.detach()

                # Calculate local prediction errors.
                # epsilon_i = activity_i - prediction_i
                for i in range(1, self.L + 1):
                    error = self.network_activations[i] - self.network_predictions[i]
                    self.network_prediction_errors[i] = clip_error(error)

                actor_error = clip_error(
                    self.actor_activation[0] - self.actor_prediction[0]
                )
                critic_error = clip_error(
                    self.critic_activation[0] - self.critic_prediction[0]
                )

                self.actor_prediction_error[0] = actor_error
                self.critic_prediction_error[0] = critic_error

                # Calculate the actor and critic activity gradients.
                # dE/dz = epsilon_actor - actor_task_signal
                # dE/dv = epsilon_critic - critic_task_signal
                actor_task_signal = self.actor_task_signal
                critic_task_signal = self.critic_task_signal

                # Match the critic signal's shape to the critic activity.
                critic_task_signal = critic_task_signal.reshape_as(
                    self.critic_activation[0]
                )

                actor_activity_gradient = actor_error - actor_task_signal
                critic_activity_gradient = critic_error - critic_task_signal

                self.actor_inference_error_signal[0] = actor_activity_gradient.detach()
                self.critic_inference_error_signal[0] = (
                    critic_activity_gradient.detach()
                )

                # Calculate the gradient for the final shared activity.
                # dE/dh = epsilon_h - J_actor^T (epsilon_actor - task_actor) - J_critic^T (epsilon_critic - task_critic)
                actor_contribution = self.actor.backward(actor_activity_gradient)
                critic_contribution = self.critic.backward(critic_activity_gradient)

                hidden_gradient = (
                    self.network_prediction_errors[self.L]
                    - actor_contribution
                    - critic_contribution
                )
                self.network_inference_error_signals[self.L] = hidden_gradient.detach()

                # Backpropagate local prediction-error contributions
                # For activity i:
                #   dE/dmu_i = epsilon_i - J_(i+1)^T epsilon_(i+1)
                for i in range(self.L - 1, 0, -1):
                    child_error = self.network_prediction_errors[i + 1]
                    child_contribution = self.network[i].backward(child_error)  # type: ignore

                    activity_gradient = (
                        self.network_prediction_errors[i] - child_contribution
                    )
                    self.network_inference_error_signals[i] = activity_gradient.detach()

                # Update ALL activities.
                # Gradients were calculated before changing any activity.
                # This avoids mixing old and new activities within a step.
                # The input activity at index 0 remains clamped.
                self.actor_activation[0] = (
                    self.actor_activation[0]
                    - self.inference_lr * actor_activity_gradient
                )
                self.critic_activation[0] = (
                    self.critic_activation[0]
                    - self.inference_lr * critic_activity_gradient
                )

                for i in range(1, self.L + 1):
                    self.network_activations[i] = (
                        self.network_activations[i]
                        - self.inference_lr * self.network_inference_error_signals[i]
                    )

    def update_weights(
        self,
        optimizer: torch.optim.Optimizer | None = None,
        error_clip: float | None = None,
    ):
        """
        Apply local predictive-coding parameter updates to all PCLayers.

        Refreshe each PCLayer's cached input using the final
        inferred activities, recomputes final predictions/errors, and then
        updates every trainable prediction function.

        optimizer=None: each PCLayer applies its direct paper-style update.
        optimizer!=None: each PCLayer assigns its negative local direction to
        parameter.grad, and this method calls the shared optimizer once.
        """
        if self.network_activations[0].numel() == 0:
            raise RuntimeError("Call initialize(observation) before parameter updates.")

        def clip_error(error: torch.Tensor) -> torch.Tensor:
            if error_clip is None:
                return error
            return error.clamp(min=-error_clip, max=error_clip)

        # Clear gradients only in optimizer mode. This ensures parameters with
        # no local gradient in this minibatch are skipped by the optimizer.
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        # Recompute predictions using FINAL inferred activities. Calling each
        # PCLayer refreshes its cached input for the subsequent local update.
        with torch.no_grad():
            for i, layer in enumerate(self.network):
                prediction = layer(self.network_activations[i])
                self.network_predictions[i + 1] = prediction.detach().clone()
                error = self.network_activations[i + 1] - prediction
                self.network_prediction_errors[i + 1] = clip_error(error).detach()

            hidden_activity = self.network_activations[self.L]
            actor_prediction = self.actor(hidden_activity)
            critic_prediction = self.critic(hidden_activity)

            self.actor_prediction[0] = actor_prediction.detach().clone()
            self.critic_prediction[0] = critic_prediction.detach().clone()
            self.actor_prediction_error[0] = clip_error(
                self.actor_activation[0] - actor_prediction
            ).detach()
            self.critic_prediction_error[0] = clip_error(
                self.critic_activation[0] - critic_prediction
            ).detach()

        for i, layer in enumerate(self.network):
            layer.update_weights(
                self.network_prediction_errors[i + 1], optimizer=optimizer
            )  # type: ignore

        self.actor.update_weights(self.actor_prediction_error[0], optimizer=optimizer)
        self.critic.update_weights(self.critic_prediction_error[0], optimizer=optimizer)

        # One shared optimizer, one step per minibatch. Never step inside a
        # PCLayer: doing so would apply different moments/timesteps per layer.
        if optimizer is not None:
            optimizer.step()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == torch.device("cuda"):
        torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.device

    envs = gym.vector.SyncVectorEnv(
        [make_env(i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(
        envs, T=args.pc_inference_steps, inference_lr=args.pc_inference_lr
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_shape = tuple(envs.single_observation_space.shape or ())
    action_shape = tuple(envs.single_action_space.shape or ())
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    optimizer_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(
                    next_obs, return_outputs=False
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = (
                torch.from_numpy(reward).to(device=device, dtype=torch.float32).view(-1)
            )
            next_obs = torch.from_numpy(next_obs).to(device=device, dtype=torch.float32)
            next_done = torch.from_numpy(next_done).to(
                device=device, dtype=torch.float32
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        approx_kl = torch.tensor(0.0, device=device)
        # ensure losses and KL stats are always defined even if no minibatch runs
        old_approx_kl = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)
        pg_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        loss = torch.tensor(0.0, device=device)

        # Anneal whichever parameter-update mechanism is active.
        lr_fraction = 1.0 - (iteration - 1.0) / args.num_iterations
        current_lr = lr_fraction * args.learning_rate
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
        else:
            direct_lr = lr_fraction * args.pc_parameter_lr
            for module in agent.modules():
                if isinstance(module, PCLayer):
                    module.lr = direct_lr

        # stop_early = False

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # PC: Forward pass
                agent.initialize(b_obs[mb_inds])

                _, newlogprob, entropy, newvalue, logits = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds], return_outputs=True
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                agent.initialize_ppo_signal(
                    ppo_loss=loss, actor_logits=logits, critic_value=newvalue
                )
                agent.inference(error_clip=args.pc_error_clip)
                agent.update_weights(optimizer=optimizer, error_clip=args.pc_error_clip)

            if args.target_kl is not None and approx_kl.item() > args.target_kl:
                # stop_early = True
                break

        # Explained variance of the critic, matching the usual PPO diagnostic.
        y_pred = b_values
        y_true = b_returns
        var_y = torch.var(y_true, unbiased=False)
        explained_var = (
            float("nan")
            if var_y.item() == 0.0
            else (1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y).item()
        )

        writer.add_scalar("charts/learning_rate", current_lr, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/total_ppo_objective", loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar(
            "losses/clipfrac",
            float(np.mean(clipfracs)) if clipfracs else 0.0,
            global_step,
        )
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        sps = int(global_step / max(time.time() - start_time, 1e-8))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(
            f"iteration={iteration}, global_step={global_step}, "
            f"ppo_objective={loss.item():.5f}, policy_loss={pg_loss.item():.5f}, "
            f"value_loss={v_loss.item():.5f}, approx_kl={approx_kl.item():.5f}, "
            f"SPS={sps},"
            # f"early_stop={stop_early}"
        )

    if args.save_model:
        save_dir = Path("runs") / run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"{args.exp_name}.pt"
        torch.save(
            {
                "model_state_dict": agent.state_dict(),
                "args": vars(args),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            },
            model_path,
        )
        print(f"Model saved to {model_path}")

    envs.close()
    writer.close()
