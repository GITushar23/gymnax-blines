import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any
import os

class EpisodeLogger:
    """Logger for tracking episode statistics."""
    
    def __init__(self, log_dir: str, env_name: str, seed: int, pseudo_reward_type: str):
        self.log_dir = log_dir
        self.env_name = env_name
        self.seed = seed
        self.pseudo_reward_type = pseudo_reward_type
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize episode tracking
        self.episode_data = []
        self.current_episodes = defaultdict(lambda: {
            'steps': 0,
            'env_reward': 0.0,
            'pseudo_reward': 0.0,
            'total_reward': 0.0
        })
        self.episode_count = 0
        
    def update_episodes(
        self, 
        env_rewards: np.ndarray,
        pseudo_rewards: np.ndarray,
        dones: np.ndarray,
        env_ids: np.ndarray = None
    ):
        """Update episode tracking with batch of environment steps."""
        batch_size = len(env_rewards)
        
        if env_ids is None:
            env_ids = np.arange(batch_size)
        
        for i, env_id in enumerate(env_ids):
            # Update current episode stats
            self.current_episodes[env_id]['steps'] += 1
            self.current_episodes[env_id]['env_reward'] += float(env_rewards[i])
            self.current_episodes[env_id]['pseudo_reward'] += float(pseudo_rewards[i])
            self.current_episodes[env_id]['total_reward'] += float(env_rewards[i] + pseudo_rewards[i])
            
            # Check if episode is done
            if dones[i]:
                # Log completed episode
                self.episode_count += 1
                episode_info = {
                    'episode_id': self.episode_count,
                    'env_id': int(env_id),
                    'steps': self.current_episodes[env_id]['steps'],
                    'env_reward': self.current_episodes[env_id]['env_reward'],
                    'pseudo_reward': self.current_episodes[env_id]['pseudo_reward'],
                    'total_reward': self.current_episodes[env_id]['total_reward']
                }
                self.episode_data.append(episode_info)
                
                # Reset episode tracking for this environment
                self.current_episodes[env_id] = {
                    'steps': 0,
                    'env_reward': 0.0,
                    'pseudo_reward': 0.0,
                    'total_reward': 0.0
                }
    
    def save_logs(self):
        """Save episode logs to CSV file."""
        if not self.episode_data:
            print("No episode data to save.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.episode_data)
        
        # Create filename with seed and pseudo reward type
        filename = f"{self.env_name}_seed{self.seed}_{self.pseudo_reward_type}_episodes.csv"
        filepath = os.path.join(self.log_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Saved episode logs to: {filepath}")
        
        # Also save summary statistics
        summary = {
            'total_episodes': len(df),
            'mean_env_reward': df['env_reward'].mean(),
            'std_env_reward': df['env_reward'].std(),
            'mean_pseudo_reward': df['pseudo_reward'].mean(),
            'std_pseudo_reward': df['pseudo_reward'].std(),
            'mean_total_reward': df['total_reward'].mean(),
            'std_total_reward': df['total_reward'].std(),
            'mean_episode_length': df['steps'].mean(),
            'std_episode_length': df['steps'].std()
        }
        
        summary_filename = f"{self.env_name}_seed{self.seed}_{self.pseudo_reward_type}_summary.txt"
        summary_filepath = os.path.join(self.log_dir, summary_filename)
        
        with open(summary_filepath, 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        return df