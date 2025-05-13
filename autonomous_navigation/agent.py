import numpy as np

class QLearningAgent:
    """Q-learning agent for autonomous navigation."""
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Q-learning agent.
        
        Args:
            n_states (int): Number of possible states
            n_actions (int): Number of possible actions
            learning_rate (float): Learning rate (alpha) for Q-value updates
            gamma (float): Discount factor for future rewards
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Rate at which epsilon decreases
            epsilon_min (float): Minimum exploration rate
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with small random values
        self.q_table = np.random.uniform(low=-0.1, high=0.1, size=(n_states, n_actions))
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy strategy.
        
        Args:
            state (int): Current state
            
        Returns:
            int: Chosen action
        """
        if np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: choose best action based on Q-values
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-value for state-action pair.
        
        Args:
            state (int): Current state
            action (int): Chosen action
            reward (float): Received reward
            next_state (int): Next state
        """
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_q_table(self, filepath):
        """
        Save Q-table to file.
        
        Args:
            filepath (str): Path to save file
        """
        try:
            np.save(filepath, self.q_table)
        except Exception as e:
            print(f"Error saving Q-table: {e}")
    
    def load_q_table(self, filepath):
        """
        Load Q-table from file.
        
        Args:
            filepath (str): Path to load file
        """
        try:
            self.q_table = np.load(filepath)
            if self.q_table.shape != (self.n_states, self.n_actions):
                raise ValueError("Loaded Q-table has incorrect dimensions")
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            # Initialize new Q-table if loading fails
            self.q_table = np.random.uniform(low=-0.1, high=0.1, size=(self.n_states, self.n_actions))
    
    def get_best_action(self, state):
        """
        Get best action for a state (no exploration).
        
        Args:
            state (int): Current state
            
        Returns:
            int: Best action based on Q-values
        """
        return np.argmax(self.q_table[state])
    
    def get_q_values(self, state):
        """
        Get Q-values for all actions in a state.
        
        Args:
            state (int): Current state
            
        Returns:
            numpy.ndarray: Array of Q-values for each action
        """
        return self.q_table[state]
