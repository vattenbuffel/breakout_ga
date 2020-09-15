# TODO:


import pygame
from player_file import Player
from ball_file import Ball
from break_able_blocks_file import Breakable_blocks
from game_file import Game
import ai_file

display_width = 640
display_height = 480
game_display = pygame.display.set_mode((display_width, display_height))

player = Player(display_width, display_height)
ball = Ball(display_width, display_height)
breakable_blocks = Breakable_blocks(display_width, display_height)
game = Game(player, ball, breakable_blocks, display_width, display_height, game_display)

# This is for human player
while 1:
    game.update(False)



