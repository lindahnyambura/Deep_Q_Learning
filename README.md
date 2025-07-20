# Deep Q Learning - Formative 2 Assignment

This project demonstrates training and evaluating a Deep Q-Network (DQN) agent using Stable Baselines3 and Gymnasium on an Atari environment.

## Environment Selection
- **Atari Environment:** Select any Atari environment from the [Gymnasium Atari collection](https://gymnasium.farama.org/environments/atari/).
- The same environment must be used for both training and evaluation.

## Project Structure
- `train.py`: Script to train a DQN agent and save its policy network.
- `play.py`: Script to load the trained model and play the game using the agent.
- `models/`: Directory where trained models are saved.

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
| Baseline | 1e-4          | 0.99  | 64    | 1.0           | 0.01        | 0.1           | 4          | 100,000         | 100,000     | 1000           |                                         |             |
| Lindah   | 5e-5          | 0.995 | 128   | 1.0           | 0.05        | 0.2           | 8          | 100,000         | 200,000     | 5000           | Added gradient clipping (max_grad_norm=10) |             |
| Sam      | 5e-4          | 0.95  | 64    | 1.0           | 0.01        | 0.1           | 4          | 100,000         | 100,000     | 1000           | Increased learning rate and improved    | 545.26      |
| Josiane  | 1e-4          | 0.99  | 128   | 1.0           | 0.05        | 0.2           | 8          | 100,000         | 200,000     | 5000           | Improved performance                    | 722         |

> Fill in the table above as you experiment with different hyperparameter sets.

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

**To view training progress in TensorBoard:**
```bash
tensorboard --logdir ./logs/
```

### Playing
```bash
python play.py
```

**Evaluation videos** are saved in the `./eval_videos/` directory. 

## Notes
- Ensure the environment name is consistent in both scripts.
- The `models/` directory will contain your trained models.
- For best results, experiment with both `MLPPolicy` and `CnnPolicy` and document your findings.

## References
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Atari Environments](https://gymnasium.farama.org/environments/atari/)