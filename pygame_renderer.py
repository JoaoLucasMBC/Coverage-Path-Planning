import pygame
import numpy as np
from coverage_env import CoverageEnv

class PygameRenderer:
    def __init__(self, env: CoverageEnv, tile_size=50):
        pygame.init()
        self.env = env
        self.H, self.W = env.H, env.W
        self.tile_size = tile_size
        self.screen = pygame.display.set_mode((self.W * tile_size, self.H * tile_size))
        pygame.display.set_caption("CoverageEnv Visualizer")
        self.clock = pygame.time.Clock()

        # Colors
        self.bg_color = (255, 255, 255)
        self.border_color = (0, 0, 0)
        self.block_color = (139, 69, 19)
        self.agent_color = (255, 0, 0)
        self.target_color = (0, 255, 0)

    def render(self):
        self.screen.fill(self.bg_color)

        for i in range(self.H):
            for j in range(self.W):
                x = j * self.tile_size
                y = i * self.tile_size
                rect = pygame.Rect(x, y, self.tile_size, self.tile_size)

                # Draw base tile: block, target, or background
                if self.env.grid[i, j] == 1:
                    pygame.draw.rect(self.screen, self.block_color, rect)
                elif (i, j) in self.env.targets:
                    pygame.draw.rect(self.screen, self.target_color, rect)
                else:
                    pygame.draw.rect(self.screen, self.bg_color, rect)

                # Overlay visited cells with semi-transparent gray (except agent's position)
                if (i, j) in self.env.visited and (i, j) != self.env.agent_pos:
                    overlay = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
                    overlay.fill((50, 50, 50, 100))  # semi-transparent gray
                    self.screen.blit(overlay, (x, y))

                # Draw agent as a smaller red square on top
                if (i, j) == self.env.agent_pos:
                    inset = int(self.tile_size * 0.2)
                    agent_rect = pygame.Rect(x + inset, y + inset,
                                            self.tile_size - 2 * inset,
                                            self.tile_size - 2 * inset)
                    pygame.draw.rect(self.screen, self.agent_color, agent_rect)

                # Draw cell border
                pygame.draw.rect(self.screen, self.border_color, rect, 1)

        pygame.display.flip()
        self.clock.tick(10)
        pygame.time.wait(150)




    def close(self):
        pygame.quit()
