import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from environment import Environment
from agent import QLearningAgent
import time

class Visualizer:
    """Handles visualization of the environment and agent."""
    
    def __init__(self, env, agent=None):
        """
        Initialize visualizer.
        
        Args:
            env (Environment): The environment to visualize
            agent (QLearningAgent, optional): The trained agent
        """
        self.env = env
        self.agent = agent
        
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title('Autonomous Navigation Simulation')
        
        # Color mapping
        self.colors = {
            0: 'white',    # Empty cell
            1: 'black',    # Obstacle
            2: 'red',      # Goal
            3: 'green',    # Agent
            4: 'blue'      # Path
        }
        
        # Initialize animation
        self.anim = None
        self.path_cells = []
    
    def _draw_grid(self, grid):
        """Draw the grid with colors."""
        self.ax.clear()
        
        # Draw cells
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                cell_value = grid[i, j]
                color = self.colors[cell_value]
                self.ax.add_patch(
                    patches.Rectangle(
                        (j, self.env.grid_size - 1 - i),
                        1, 1,
                        facecolor=color,
                        edgecolor='gray',
                        linewidth=1
                    )
                )
        
        # Draw path cells
        for cell in self.path_cells:
            if cell != self.env.goal_pos:  # Don't draw over goal
                self.ax.add_patch(
                    patches.Rectangle(
                        (cell[1], self.env.grid_size - 1 - cell[0]),
                        1, 1,
                        facecolor='lightblue',
                        edgecolor='gray',
                        alpha=0.5
                    )
                )
        
        # Set grid properties
        self.ax.set_xlim(-0.5, self.env.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.env.grid_size - 0.5)
        self.ax.set_xticks(range(self.env.grid_size))
        self.ax.set_yticks(range(self.env.grid_size))
        self.ax.grid(True)
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor='white', edgecolor='gray', label='Empty'),
            patches.Patch(facecolor='black', label='Obstacle'),
            patches.Patch(facecolor='red', label='Goal'),
            patches.Patch(facecolor='green', label='Agent'),
            patches.Patch(facecolor='lightblue', alpha=0.5, label='Path')
        ]
        self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
    
    def _update(self, frame):
        """Update function for animation."""
        if self.agent is None:
            return
        
        state = self.env._get_state()
        action = self.agent.get_best_action(state)
        next_state, reward, done, _ = self.env.step(action)
        
        # Add current position to path
        self.path_cells.append(self.env.current_pos)
        
        # Draw updated grid
        self._draw_grid(self.env.render())
        
        if done:
            self.anim.event_source.stop()
            plt.title("Goal Reached!", pad=20)
        
        return self.ax,
    
    def show_static(self):
        """Show a static view of the environment."""
        self._draw_grid(self.env.render())
        plt.show()
    
    def animate(self, interval=500):
        """
        Animate the agent's movement through the environment.
        
        Args:
            interval (int): Time between frames in milliseconds
        """
        self.path_cells = []
        self.env.reset()
        
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            frames=None,
            interval=interval,
            blit=True,
            repeat=False
        )
        
        plt.show()
    
    def plot_training_metrics(self, episode_rewards, episode_lengths):
        """
        Plot training metrics.
        
        Args:
            episode_rewards (list): Rewards per episode
            episode_lengths (list): Steps per episode
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot rewards
        ax1.plot(episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        
        # Plot episode lengths
        ax2.plot(episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main visualization function."""
    # Create environment and agent
    env = Environment(grid_size=10)
    agent = QLearningAgent(env.n_states, env.n_actions)
    
    # Try to load trained Q-table
    try:
        agent.load_q_table("models/q_table.npy")
        print("Loaded trained Q-table")
    except:
        print("No trained Q-table found. Agent will act randomly.")
    
    # Create visualizer and run animation
    vis = Visualizer(env, agent)
    vis.animate(interval=500)

if __name__ == "__main__":
    main()
