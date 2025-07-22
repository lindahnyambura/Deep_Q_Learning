import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo
import os
import subprocess
import ale_py

# Configuration for evaluation
CONFIG = {
    "MODEL_PATH": "models/best_model.zip",  # Path to the trained model
    "ENV_ID": "ALE/Assault-v5",  # Atari environment ID
    "NUM_EPISODES": 5,  # Number of episodes to evaluate
    "FRAME_STACK": 4,  # Number of frames to stack (must match training)
    "VIDEO_DIR": "./eval_videos/",  # Directory to save evaluation videos
    "MAX_STEPS": 1000  # Max steps per episode
}

def create_env():
    """
    Create and return an Atari environment with preprocessing, frame stacking, and video recording.
    """
    # Create the base Atari environment
    env = gym.make(
        CONFIG["ENV_ID"],
        render_mode="rgb_array",
        frameskip=1,
        full_action_space=False
    )

    # Apply Atari preprocessing (grayscale, resize, frame skip, etc.)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True
    )

    # Stack frames to provide temporal context to the agent
    env = FrameStackObservation(env, CONFIG["FRAME_STACK"])

    # Ensure the video directory exists
    os.makedirs(CONFIG["VIDEO_DIR"], exist_ok=True)

    # Wrap the environment to record videos of each episode
    env = RecordVideo(
        env,
        CONFIG["VIDEO_DIR"],
        episode_trigger=lambda x: True,  # Record every episode
        name_prefix="eval"
    )

    return env

def run_evaluation(model, env):
    """Run evaluation with video recording"""
    print("\n=== Running Evaluation ===")
    rewards = []

    for ep in range(CONFIG["NUM_EPISODES"]):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done and step < CONFIG["MAX_STEPS"]:
            # Convert observation to model input format
            # FrameStackObservation returns (4,84,84) array
            obs_tensor = np.expand_dims(obs, 0)  # Add batch dim -> (1,4,84,84)
            action, _ = model.predict(obs_tensor, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action[0])
            done = terminated or truncated
            total_reward += reward
            step += 1

        rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward={total_reward}, Steps={step}")

    print("\n=== Final Results ===")
    print(f"Mean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Min/Max: {np.min(rewards)}/{np.max(rewards)}")

    # Display videos in Colab
    video_paths = sorted([
        f for f in os.listdir(CONFIG["VIDEO_DIR"])
        if f.endswith(".mp4")
    ])[:CONFIG["NUM_EPISODES"]]

    if video_paths:
        print("\nRecorded Videos:")
        for vid in video_paths:
            video_path = os.path.join(CONFIG["VIDEO_DIR"], vid)
            print(f"Video saved at: {video_path}")

if __name__ == "__main__":
    print("=== Environment Setup ===")
    env = create_env()
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    print("\n=== Model Setup ===")
    model = DQN.load(CONFIG["MODEL_PATH"])

    print("\n=== Starting Evaluation ===")
    run_evaluation(model, env)
    env.close()

    # Create zip for download
    subprocess.run([
        "zip", "-r", "eval_videos.zip", CONFIG["VIDEO_DIR"]
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("\n=== Evaluation Complete ===")
    print("Download videos with:")
    print("eval_videos.zip created in the current directory.")
