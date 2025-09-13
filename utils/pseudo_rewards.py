import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional
from functools import partial
import numpy as np

class PseudoRewardWrapper:
    """Wrapper to add pseudo rewards to the environment rewards based on the BAMPF paper."""
    
    def __init__(
        self, 
        pseudo_reward_type: str, 
        num_envs: int,
        gamma: float = 0.99,
        max_episode_steps: int = 200
    ):
        self.pseudo_reward_type = pseudo_reward_type
        self.num_envs = num_envs
        self.gamma = gamma
        self.max_episode_steps = max_episode_steps
        
        # Scaling factors from the paper
        self.SCALING_FACTOR_DISPLACEMENT = 4.0
        self.SCALING_FACTOR_PBS = 10.0
        self.SCALING_FACTOR_BAMPF = 10.0
        
        # Initialize persistent state for each environment
        self.reset_persistent_state()
        
    def reset_persistent_state(self):
        """Reset persistent state for all environments."""
        # For PBS: track previous potential
        self.potential_pbs_prev = np.zeros(self.num_envs)
        
        # For BAMPF: track previous potential and max displacement
        self.potential_bampf_prev = np.zeros(self.num_envs)
        self.max_disp_so_far = np.zeros(self.num_envs)
        
        # Track episode steps for BAMPF termination handling
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int32)
    
    def compute_pseudo_reward(
        self, 
        obs: jnp.ndarray, 
        action: jnp.ndarray, 
        next_obs: jnp.ndarray,
        done: jnp.ndarray,
        info: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute pseudo reward with persistent state handling."""
        
        # Convert JAX arrays to numpy for computation
        obs_np = np.array(obs)
        next_obs_np = np.array(next_obs)
        done_np = np.array(done)
        
        batch_size = len(done_np)
        pseudo_rewards = np.zeros(batch_size, dtype=np.float32)
        
        if self.pseudo_reward_type == "none":
            pass  # pseudo_rewards already zeros
            
        elif self.pseudo_reward_type == "displacement":
            # Simple displacement reward (no persistent state needed)
            displacement = np.abs(next_obs_np[:, 0] - (-0.5))
            pseudo_rewards = self.SCALING_FACTOR_DISPLACEMENT * displacement
            
        elif self.pseudo_reward_type == "displacement_pbs":
            # Potential-based shaping with displacement potential
            for i in range(batch_size):
                # Current potential
                potential_current = np.abs(next_obs_np[i, 0] - (-0.5))
                
                # If episode is done, potential is 0
                if done_np[i]:
                    potential_current = 0.0
                
                # PBS reward: gamma * phi(s') - phi(s)
                r_intrinsic_pbs = self.gamma * potential_current - self.potential_pbs_prev[i]
                pseudo_rewards[i] = self.SCALING_FACTOR_PBS * r_intrinsic_pbs
                
                # Update stored potential
                self.potential_pbs_prev[i] = potential_current
                
                # Reset episode steps counter if done
                if done_np[i]:
                    self.episode_steps[i] = 0
                else:
                    self.episode_steps[i] += 1
                    
        elif self.pseudo_reward_type == "max_displacement_bampf":
            # BAMPF: Exponentially smoothed maximum displacement
            ALPHA = 0.5
            
            for i in range(batch_size):
                # Current displacement
                current_displacement = np.abs(next_obs_np[i, 0] - (-0.5))
                
                # Update max displacement seen so far
                self.max_disp_so_far[i] = max(self.max_disp_so_far[i], current_displacement)
                
                # Exponentially smoothed maximum (from paper formula)
                # Mt = 0.5 * max(Mt-1, disp) + 0.5 * Mt-1
                potential_current = (0.5 * max(self.potential_bampf_prev[i], current_displacement) + 
                                   0.5 * self.potential_bampf_prev[i])
                
                # Handle episode termination
                if done_np[i]:
                    # Apply discount factor based on remaining steps
                    te = self.episode_steps[i]  # Current timestep in episode
                    potential_current *= (self.gamma ** (self.max_episode_steps - te))
                
                # BAMPF reward: gamma * phi(h') - phi(h)
                r_intrinsic_bampf = self.gamma * potential_current - self.potential_bampf_prev[i]
                pseudo_rewards[i] = self.SCALING_FACTOR_BAMPF * r_intrinsic_bampf
                
                # Update stored potential
                self.potential_bampf_prev[i] = potential_current
                
                # Reset episode steps counter and max displacement if done
                if done_np[i]:
                    self.episode_steps[i] = 0
                    # Note: We don't reset max_disp_so_far as it tracks across episodes
                else:
                    self.episode_steps[i] += 1
        
        else:
            raise ValueError(f"Unknown pseudo reward type: {self.pseudo_reward_type}")
        
        # Increment episode steps for all environments
        # (This is redundant for PBS and BAMPF but needed for other types)
        if self.pseudo_reward_type in ["none", "displacement"]:
            self.episode_steps += 1
            self.episode_steps[done_np] = 0
        
        # Return as JAX array
        return jnp.array(pseudo_rewards)