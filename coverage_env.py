import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CoverageEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, curriculum=0, max_steps=200, seed=None):
        super().__init__()
        self.H, self.W = 8, 8
        self.max_steps = max_steps
        self.curriculum = curriculum

        # seeding for reproducibility
        self.seed(seed)

        # Action: down, up, right, left
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4, self.H, self.W), dtype=np.float32)

    def seed(self, seed=None):
        """
        Seed the environment's RNGs for reproducible layouts.
        """
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment; returns obs (shape C×H×W), info
        Channels:
         0 = free space
         1 = obstacles
         2 = agent location
         3 = target area
         4 = visited mask (all zeros at reset)
        """

        if seed is not None:
            self.seed(seed)
        else:
            self.seed()

        # Read curriculum txt from the self.curriculum variable
        with open(f"levels/curriculum_{self.curriculum}.txt", "r") as f:
            lines = f.readlines()
        
        # # -> obstacle
        # . -> free space
        # T -> target

        grid = np.zeros((self.H, self.W), dtype=np.int8)
        targets = []
        for i, line in enumerate(lines):
            line = line.strip()
            for j, c in enumerate(line):
                if c == "#":
                    grid[i, j] = 1
                elif c == "T":
                    targets.append((i, j))

        while True:
            # Determine a valid start position
            start = (random.randint(0, self.H-1), random.randint(0, self.W-1))
            if grid[start] == 0 and start not in targets:
                break

        self.grid = grid
        self.targets = set(targets)
        self.agent_pos = start
        self.visited = { start }
        self.steps = 0

        return self._get_obs(), {}

    def _get_obs(self):
        C = 4
        state = np.zeros((C, self.H, self.W), dtype=np.float32)

        # channel 0: free space (grid==0)
        state[0, :, :] = (self.grid == 0).astype(np.float32)
        # channel 2: agent location
        i, j = self.agent_pos
        state[1, i, j] = 1.0
        # channel 3: target area
        for (ti, tj) in self.targets:
            state[2, ti, tj] = 1.0
        for vi, vj in self.visited:
            state[3, vi, vj] = 1.0
        
        return state

    def step(self, action):
        # Define movement vectors
        # 0 = down, 1 = up, 2 = right, 3 = left
        moves = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
        i, j = self.agent_pos
        di, dj = moves[action]
        ni, nj = i + di, j + dj

        # Default baseline reward
        reward = -1

        # Check validity and apply penalties
        if not (0 <= ni < self.H and 0 <= nj < self.W and self.grid[ni, nj] == 0):
            # Invalid action: stay in place
            self.agent_pos = (i, j) 
        else:
            # Valid move: update position
            self.agent_pos = (ni, nj)

        # Check if on a target
        if self.agent_pos not in self.visited:
            if self.agent_pos in self.targets:
                reward = 2   # new target
            self.visited.add(self.agent_pos)

        # Step count
        self.steps += 1

        # Terminal bonus
        terminated = True
        for (i,j) in self.targets:
            if (i,j) not in self.visited:
                terminated = False
                break
            
        truncated = (self.steps >= self.max_steps)
        if terminated:
            reward += 30.0

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode="human"):
        disp = np.full((self.H, self.W), '.', dtype=str)
        for (i,j) in self.targets:
            disp[i,j] = 'T'
        for (i,j) in zip(*np.where(self.grid == 1)):
            disp[i,j] = '#'
        ai, aj = self.agent_pos
        disp[ai, aj] = 'A'
        print("\n".join("".join(row) for row in disp))

    def close(self):
        pass