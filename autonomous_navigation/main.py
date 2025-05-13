import argparse
import logging
from pathlib import Path
import sys
import time

from environment import Environment
from agent import QLearningAgent
from train import Trainer
from visualize import Visualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Autonomous Navigation System using Q-Learning'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the agent'
    )
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Run simulation with trained agent'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=500,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=10,
        help='Size of the grid environment'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=200,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/q_table.npy',
        help='Path to save/load Q-table'
    )
    
    return parser.parse_args()

def train_agent(args):
    """Train the agent and save Q-table."""
    logger.info("Initializing training...")
    
    # Create trainer
    trainer = Trainer(
        grid_size=args.grid_size,
        n_episodes=args.episodes,
        max_steps=args.max_steps
    )
    
    # Train agent
    metrics = trainer.train(q_table_path=args.model_path)
    
    # Visualize training metrics
    vis = Visualizer(trainer.env)
    vis.plot_training_metrics(
        metrics['episode_rewards'],
        metrics['episode_lengths']
    )
    
    logger.info(f"Training completed. Q-table saved to {args.model_path}")
    return trainer.env, trainer.agent

def run_simulation(args):
    """Run simulation with trained agent."""
    logger.info("Initializing simulation...")
    
    # Create environment and agent
    env = Environment(grid_size=args.grid_size)
    agent = QLearningAgent(env.n_states, env.n_actions)
    
    # Try to load trained Q-table
    try:
        agent.load_q_table(args.model_path)
        logger.info("Loaded trained Q-table")
    except Exception as e:
        logger.error(f"Error loading Q-table: {e}")
        logger.warning("Agent will act randomly!")
    
    # Create visualizer and run animation
    vis = Visualizer(env, agent)
    vis.animate(interval=500)
    
    return env, agent

def main():
    """Main function."""
    args = parse_args()
    
    # Create models directory if it doesn't exist
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    
    if not args.train and not args.simulate:
        logger.error("Please specify either --train or --simulate")
        sys.exit(1)
    
    try:
        if args.train:
            env, agent = train_agent(args)
            
            # After training, run a simulation if requested
            if args.simulate:
                logger.info("Training complete. Starting simulation...")
                time.sleep(2)  # Brief pause between training and simulation
                run_simulation(args)
        
        elif args.simulate:
            run_simulation(args)
    
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
