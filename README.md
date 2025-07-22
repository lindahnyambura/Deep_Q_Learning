# Deep Q Learning - Formative 2 Assignment

This project demonstrates training and evaluating a Deep Q-Network (DQN) agent using Stable Baselines3 and Gymnasium on an Atari environment.

## Environment Selection
- **Atari Environment:** Select any Atari environment from the [Gymnasium Atari collection](https://gymnasium.farama.org/environments/atari/).
- The same environment must be used for both training and evaluation.

## Project Structure
- `train.py`: Script to train a DQN agent and save its policy network.
- `play.py`: Script to load the trained model and play the game using the agent.
- `models/`: Directory where trained models are saved.
- `eval_videos/`: Directory where evaluation videos and gifs are saved.

## Requirements
- Python 3.7+
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [gymnasium[atari]](https://gymnasium.farama.org/environments/atari/)
- [torch](https://pytorch.org/)

Install dependencies:
```bash
pip install stable-baselines3[extra] gymnasium[atari] torch
```

## 1. Training the Agent (`train.py`)

- Defines and trains a DQN agent using Stable Baselines3.
- Compares `MLPPolicy` (Multilayer Perceptron) and `CnnPolicy` (Convolutional Neural Network) for the chosen environment.
- Logs key training details: reward trends, episode length, and hyperparameters.
- Saves the trained model as `models/dqn_model.zip`.

### Hyperparameter Tuning
Tune and document the following hyperparameters:
- **Learning Rate (lr)**
- **Gamma (γ)**: Discount factor
- **Batch Size (batch_size)**
- **Epsilon (epsilon_start, epsilon_end, epsilon_decay)**: Controls exploration in ε-greedy policies

#### Hyperparameter Set Documentation

| Name     | Learning Rate | Gamma | Batch | Epsilon Start | Epsilon End | Epsilon Decay | Train Freq | Total Timesteps | Buffer Size | Target Interval | Noted Behaviour                        | Mean Reward |
|----------|---------------|-------|-------|---------------|-------------|---------------|------------|-----------------|-------------|----------------|-----------------------------------------|-------------|
| Baseline | 1e-4          | 0.99  | 64    | 1.0           | 0.01        | 0.1           | 4          | 100,000         | 100,000     | 1000           |       Faster initial learning but unstable, with reward collapse (-260 from peak)                                  |       441     |
| Lindah   | 5e-5          | 0.995 | 128   | 1.0           | 0.05        | 0.2           | 8          | 100,000         | 200,000     | 5000           | Added gradient clipping (max_grad_norm=10) Slower start but more stable, higher final reward (+58.5%) with no collapse.|        699     |
| Sam      | 5e-4          | 0.95  | 64    | 1.0           | 0.01        | 0.1           | 4          | 100,000         | 500,000     | 1000           | Increased learning rate and buffer size and improved    | 554.26      |
| Josiane  | 1e-4          | 0.99  | 128   | 1.0           | 0.05        | 0.2           | 8          | 100,000         | 200,000     | 5000           | Improved performance                    | 722         |
| Miracle   | 1e-4          | 0.99  | 64    | 1.0           | 0.02        | 0.2           | 4          | 100,000            | 200,000     | 1000           | Larger batch, buffer, more exploration, longer learning_starts, n_stack=4 |    2.90         |

> Fill in the table above as you experiment with different hyperparameter sets.

### Comparison of MlpPolicy and CnnPolicy for Assault-v5

- **Performance Superiority**: CnnPolicy consistently achieves higher final mean episode rewards than MlpPolicy across all experiments, demonstrating its effectiveness in the Assault-v5 environment.
- **Architectural Advantage**: The convolutional architecture of CnnPolicy excels at extracting spatial features from the game's visual input, making it well-suited for Atari-like tasks.
- **MlpPolicy Limitations**: MlpPolicy, based on a multi-layer perceptron, struggles to process raw pixel data, leading to lower performance compared to CnnPolicy.
- **Hyperparameter Impact**: Experiments varying learning rate, exploration duration, replay buffer size, and gamma show CnnPolicy's robustness, with the largest replay buffer yielding the best results for both policies.
- **Optimization Preference**: CnnPolicy benefits more from hyperparameter tuning, particularly with a larger replay buffer and higher learning rate, making it the preferred choice for optimizing performance in Assault-v5.

## 2. Playing with the Agent (`play.py`)

- Loads the trained model using `DQN.load('models/dqn_model.zip')`.
- Sets up the same Atari environment as in training.
- Uses **GreedyQPolicy** for evaluation (agent selects actions with highest Q-value).
- Runs and visualizes a few episodes using `env.render()`.

## Usage

### Training
```bash
python train.py
```

### Playing
```bash
python play.py
```

**Evaluation videos** are saved in the `./eval_videos/` directory. 

## Example Evaluation (GIF)

Below is an example of the agent playing the Atari environment:

![Evaluation Example](eval_videos/eval-episode.gif)

## Notes
- Ensure the environment name is consistent in both scripts.
- The `models/` directory will contain your trained models.
- For best results, experiment with both `MLPPolicy` and `CnnPolicy` and document your findings.

