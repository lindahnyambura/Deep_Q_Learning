import os
from typing import Dict, Any
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import FlattenObservation

# CONFIGS
CONFIG = {
    # Environment
    "ENV_ID": "ALE/Assault-v5",
    "TOTAL_TIMESTEPS": 100_000,
    "EVAL_FREQ": 10_000,
    "FRAME_STACK": 4,
    
    # Policy Selection (Change this only!)
    "POLICY": "MlpPolicy",  # Options: "CnnPolicy" or "MlpPolicy"
    
    # Hyperparameters (Shared)
    "HYPERPARAMS": {
        "learning_rate": 1e-4,            # start w conservative LR
        "buffer_size": 100_000,           # replay buffer size
        "batch_size": 64,                 # batch size for training
        "gamma": 0.99,                    # discount factor
        "exploration_fraction": 0.1,      # % of timesteps for exploration
        "exploration_initial_eps": 1.0,   # initial exploration rate
        "exploration_final_eps": 0.01,    # final exploration rate
        "target_update_interval": 1000,   # update target network every N steps
        "train_freq": 4,                  # update model every N steps
    }
}


# ENVIRONMENT
def create_env(env_id: str, policy_type: str, eval: bool = False) -> VecFrameStack:
    """Modified to handle MLP flattening"""
    env = make_atari_env(
        env_id,
        n_envs=1,
        monitor_dir="./logs/monitor" if eval else None
    )
    
    # MLP requires flattened observations
    if policy_type == "MlpPolicy":
        env = DummyVecEnv([lambda: FlattenObservation(env.envs[0])])
    
    return VecFrameStack(env, n_stack=CONFIG["FRAME_STACK"])


# TRAINING FUNCTION
def train_dqn() -> DQN:
    """Simplified to use CONFIG"""
    train_env = create_env(CONFIG["ENV_ID"], CONFIG["POLICY"])
    eval_env = create_env(CONFIG["ENV_ID"], CONFIG["POLICY"], eval=True)

    model = DQN(
        policy=CONFIG["POLICY"],
        env=train_env,
        verbose=1,
        tensorboard_log=f"./logs/{CONFIG['POLICY']}_tensorboard/",
        **CONFIG["HYPERPARAMS"]
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{CONFIG['POLICY']}_best/",
        log_path=f"./logs/{CONFIG['POLICY']}_eval/",
        eval_freq=CONFIG["EVAL_FREQ"],
        deterministic=True
    )

    model.learn(
        total_timesteps=CONFIG["TOTAL_TIMESTEPS"],
        callback=eval_callback,
        progress_bar=True
    )

    model.save(f"./models/dqn_{CONFIG['ENV_ID'].split('/')[-1]}_{CONFIG['POLICY']}")
    train_env.close()
    eval_env.close()
    return model

# MAIN
if __name__ == "__main__":
    # Setup directories
    os.makedirs("./models/best", exist_ok=True)
    os.makedirs("./logs/tensorboard", exist_ok=True)
    os.makedirs("./logs/evaluations", exist_ok=True)

    # Print training header
    print("\n" + "="*50)
    print(f"Training {CONFIG['POLICY']} on {CONFIG['ENV_ID']}")
    print(f"Total timesteps: {CONFIG['TOTAL_TIMESTEPS']}")
    print(f"Evaluation frequency: Every {CONFIG['EVAL_FREQ']} steps")
    print("Hyperparameters:")
    for k, v in CONFIG["HYPERPARAMS"].items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")

    # Train model
    model = train_dqn()

    # Save and final report
    model_save_path = f"./models/dqn_{CONFIG['ENV_ID'].split('/')[-1]}_{CONFIG['POLICY']}_final"
    model.save(model_save_path)
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Model saved to: {model_save_path}.zip")
    print(f"Best model during training saved to: ./models/{CONFIG['POLICY']}_best/")
    print(f"Tensorboard logs: ./logs/{CONFIG['POLICY']}_tensorboard/")
    print("="*50)