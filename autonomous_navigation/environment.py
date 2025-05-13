import numpy as np

class Environment:
    """2D grid environment for autonomous navigation."""
    
    def __init__(self, grid_size=10, obstacle_mode='random', obstacle_ratio=0.2):
        """
        Initialize the environment.
        
        Args:
            grid_size (int): Size of the square grid (default: 10)
            obstacle_mode (str): 'random' or 'predefined' (default: 'random')
            obstacle_ratio (float): Ratio of obstacles to total grid size (default: 0.2)
        """
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size-1, grid_size-1)
        self.current_pos = self.start_pos
        
        # Place obstacles
        if obstacle_mode == 'random':
            self._place_random_obstacles(obstacle_ratio)
        
        # Mark goal position
        self.grid[self.goal_pos] = 2
    
    def _place_random_obstacles(self, ratio):
        """Place random obstacles in the grid."""
        n_obstacles = int(self.grid_size * self.grid_size * ratio)
        count = 0
        while count < n_obstacles:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            # Don't place obstacles at start or goal
            if (x, y) != self.start_pos and (x, y) != self.goal_pos:
                if self.grid[x, y] == 0:  # If cell is empty
                    self.grid[x, y] = 1  # Place obstacle
                    count += 1
    
    def reset(self):
        """Reset agent to start position."""
        self.current_pos = self.start_pos
        return self._get_state()
    
    def _get_state(self):
        """Convert current position to state number."""
        return self.current_pos[0] * self.grid_size + self.current_pos[1]
    
    def step(self, action):
        """
        Take action and return new state, reward, and done flag.
        
        Args:
            action (int): 0=up, 1=right, 2=down, 3=left
            
        Returns:
            tuple: (new_state, reward, done, info)
        """
        x, y = self.current_pos
        
        # Calculate new position based on action
        if action == 0:  # up
            new_pos = (max(0, x-1), y)
        elif action == 1:  # right
            new_pos = (x, min(self.grid_size-1, y+1))
        elif action == 2:  # down
            new_pos = (min(self.grid_size-1, x+1), y)
        else:  # left
            new_pos = (x, max(0, y-1))
        
        # Check if new position is valid
        if self.grid[new_pos] == 1:  # Hit obstacle
            reward = -10
            done = False
            new_pos = self.current_pos  # Stay in current position
        elif new_pos == self.goal_pos:  # Reached goal
            reward = 100
            done = True
        else:  # Valid move
            reward = -1  # Small penalty for each move to encourage efficiency
            done = False
        
        self.current_pos = new_pos
        return self._get_state(), reward, done, {}
    
    def render(self):
        """
        Return a copy of the grid with current position marked.
        
        Returns:
            numpy.ndarray: Grid with current position marked as 3
        """
        render_grid = self.grid.copy()
        if self.current_pos != self.goal_pos:  # Don't overwrite goal position
            render_grid[self.current_pos] = 3
        return render_grid
    
    def is_valid_move(self, position):
        """
        Check if a position is valid (within bounds and not an obstacle).
        
        Args:
            position (tuple): (x, y) coordinates to check
            
        Returns:
            bool: True if position is valid, False otherwise
        """
        x, y = position
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.grid[position] != 1
        return False
    
    @property
    def n_states(self):
        """Total number of possible states."""
        return self.grid_size * self.grid_size
    
    @property
    def n_actions(self):
        """Number of possible actions."""
        return 4  # up, right, down, left
