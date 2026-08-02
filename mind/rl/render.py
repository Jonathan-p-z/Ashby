"""Live top-down view of the human-vs-Ashby recording session.

record.py owns the gamepad and the sim step loop; this module only owns a pygame
window and draws whatever GameState it's handed each step. Kept separate so a
render hiccup is a render bug, never a training-data bug -- record.py's capture
loop doesn't know or care how (or whether) the window is drawing.

No 3D, no car models, no camera -- just enough of a map to tell who's got the ball
and who's about to run dry on boost.
"""

import os
import sys

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(__file__))
from rlgym_sim.utils import common_values
from rlgym_sim.utils.gamestates import GameState

WINDOW_W = 600
WINDOW_H = 400
HUD_H = 40  # top strip for score + boost bars; the rest below it is the pitch

FIELD_X = common_values.SIDE_WALL_X  # +-4096, world units
FIELD_Y = common_values.BACK_WALL_Y  # +-5120, world units

BG_COLOR = (18, 40, 22)
LINE_COLOR = (70, 110, 75)
HOME_COLOR = (60, 130, 255)  # you
AWAY_COLOR = (235, 70, 70)   # Ashby
BALL_COLOR = (240, 240, 240)
TEXT_COLOR = (230, 230, 230)
BAR_BG = (60, 60, 60)

CAR_PX = 10   # car marker size in screen pixels -- not to scale, just visible
BALL_PX = 5
BAR_W, BAR_H = 150, 10


def _world_to_screen(x: float, y: float) -> tuple:
    """Maps the ground plane (+-4096 x, +-5120 y) onto the pitch strip below the HUD."""
    sx = (x + FIELD_X) / (2 * FIELD_X) * WINDOW_W
    sy = HUD_H + (y + FIELD_Y) / (2 * FIELD_Y) * (WINDOW_H - HUD_H)
    return int(sx), int(sy)


class MatchRenderer:
    """Owns the pygame display window. One instance per recording session."""

    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Ashby -- Live Match")
        self.font = pygame.font.SysFont("consolas", 20)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def update(self, state: GameState) -> bool:
        """Draws one frame from a GameState snapshot. Returns False once the window
        has been closed, so record.py knows to stop the session cleanly instead of
        driving a dead window every step."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed = True

        if self._closed:
            return False

        home, away = state.players[0], state.players[1]

        self._draw_field()
        self._draw_car(home.car_data.position, HOME_COLOR)
        self._draw_car(away.car_data.position, AWAY_COLOR)
        self._draw_ball(state.ball.position)
        self._draw_bar(10, 26, home.boost_amount, (60, 220, 90))
        self._draw_bar(WINDOW_W - 10 - BAR_W, 26, away.boost_amount, (220, 70, 70))
        self._draw_score(state.blue_score, state.orange_score)

        pygame.display.flip()
        return True

    def close(self):
        pygame.display.quit()

    def _draw_field(self):
        self.screen.fill(BG_COLOR)
        pitch = pygame.Rect(0, HUD_H, WINDOW_W, WINDOW_H - HUD_H)
        pygame.draw.rect(self.screen, LINE_COLOR, pitch, width=2)
        mid_y = HUD_H + (WINDOW_H - HUD_H) // 2
        pygame.draw.line(self.screen, LINE_COLOR, (0, mid_y), (WINDOW_W, mid_y), 1)
        pygame.draw.circle(self.screen, LINE_COLOR, (WINDOW_W // 2, mid_y), 28, 1)

    def _draw_car(self, pos, color):
        cx, cy = _world_to_screen(pos[0], pos[1])
        rect = pygame.Rect(0, 0, CAR_PX, CAR_PX)
        rect.center = (cx, cy)
        pygame.draw.rect(self.screen, color, rect)

    def _draw_ball(self, pos):
        cx, cy = _world_to_screen(pos[0], pos[1])
        pygame.draw.circle(self.screen, BALL_COLOR, (cx, cy), BALL_PX)

    def _draw_bar(self, x: int, y: int, pct: float, color):
        pygame.draw.rect(self.screen, BAR_BG, (x, y, BAR_W, BAR_H))
        fill_w = int(BAR_W * float(np.clip(pct, 0.0, 1.0)))
        pygame.draw.rect(self.screen, color, (x, y, fill_w, BAR_H))
        pygame.draw.rect(self.screen, TEXT_COLOR, (x, y, BAR_W, BAR_H), width=1)

    def _draw_score(self, blue_score: int, orange_score: int):
        surf = self.font.render(f"YOU {blue_score} - {orange_score} ASHBY", True, TEXT_COLOR)
        rect = surf.get_rect(center=(WINDOW_W // 2, 14))
        self.screen.blit(surf, rect)
