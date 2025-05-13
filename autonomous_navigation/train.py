import numpy as np
from environment import Environment
from agent import QLearningAgent
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Trainer:
    """Handles the training of the Q-learning agent."""
    
    def __init__(self, grid_size=10, n_episodes=500, max_steps=200):
        """
        Initialize trainer.
        
        Args:
            grid_size (int): Size of the grid environment
            n_episodes (int): Number of training episodes
            max_steps (int): Maximum steps per episode
        """
        self.env = Environment(grid_size=grid_size)
        self.agent = QLearningAgent(
            n_states=self.env.n_states,
            n_actions=self.env.n_actions
        )
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
    
    def train(self, q_table_path="q_table.npy"):
        """
        Train the agent.
        
        Args:
            q_table_path (str): Path to save the trained Q-table
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        for episode in range(self.n_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps):
                # Choose and take action
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Update Q-table
                self.agent.update(state, action, reward, next_state)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            
            # Log progress
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_length = np.mean(self.episode_lengths[-50:])
                logger.info(
                    f"Episode {episode + 1}/{self.n_episodes} - "
                    f"Avg Reward: {avg_reward:.2f} - "
                    f"Avg Length: {avg_length:.2f} - "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )
        
        # Save trained Q-table
        Path(q_table_path).parent.mkdir(parents=True, exist_ok=True)
        self.agent.save_q_table(q_table_path)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_time': training_time
        }
    
    def test_episode(self, render=True):
        """
        Run a test episode with the trained agent.
        
        Args:
            render (bool): Whether to render the environment
            
        Returns:
            tuple: (total_reward, steps_taken, path)
        """
        state = self.env.reset()
        total_reward = 0
        path = [self.env.current_pos]
        
        for step in range(self.max_steps):
            if render:
                print(self.env.render())
            
            # Use best action (no exploration)
            action = self.agent.get_best_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            total_reward += reward
            path.append(self.env.current_pos)
            state = next_state
            
            if done:
                if render:
                    print(self.env.render())
                break
        
        return total_reward, step + 1, path

def main():
    """Main training function."""
    # Training parameters
    GRID_SIZE = 10
    N_EPISODES = 500
    MAX_STEPS = 200
    Q_TABLE_PATH = "models/q_table.npy"
    
    # Create and train agent
    trainer = Trainer(
        grid_size=GRID_SIZE,
        n_episodes=N_EPISODES,
        max_steps=MAX_STEPS
    )
    
    # Train the agent
    metrics = trainer.train(q_table_path=Q_TABLE_PATH)
    
    # Run a test episode
    logger.info("Running test episode...")
    reward, steps, path = trainer.test_episode(render=True)
    logger.info(f"Test episode - Reward: {reward}, Steps: {steps}")
    
    return metrics

if __name__ == "__main__":
    main()
