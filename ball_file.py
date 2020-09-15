import pygame
from square_file import Square
import math
import numpy as np


# TODO:


class Ball():
    def __init__(self, display_width, display_height):
        self.x_start = 100
        self.y_start = 300
        self.dx = 5
        self.dy = 5
        self.ball_width = 20
        self.ball_height = 20
        self.angle_increase = math.radians(45)
        self.angle_max = math.radians(150)

        self.angle = math.pi / 4
        self.color = (0, 0, 255)  # blue
        self.display_width = display_width
        self.display_height = display_height
        self.ball = Square(self.x_start, self.y_start, self.ball_width, self.ball_height, self.color)
        self.out_of_bound = False

    def update(self, player, break_able_blocks):  # Maybe do this recursively....
        # Calc next position
        x_new = self.ball.x + self.dx * abs(math.cos(self.angle))
        y_new = self.ball.y + self.dy * math.sin(self.angle)

        # Check for collision with walls
        if (x_new + self.ball_width > self.display_width) or (x_new < 0):
            self.dx = -self.dx
            x_new = self.ball.x + self.dx * abs(math.cos(self.angle))

        if (y_new < 0):
            self.dy = -self.dy
            y_new = self.ball.y + self.dy * math.sin(self.angle)

        # Check for out of bound
        if (y_new + self.ball_height > self.display_height):
            self.out_of_bound = True

        # Check for collision with player
        if (y_new + self.ball_height) >= player.plate_y_start:
            # The ball is on the appropriate height to bounce
            plate_left = player.plate[0].x
            plate_right = player.plate[-1].x + player.plate_square_width
            ball_left = x_new
            ball_right = x_new + self.ball_width

            if (ball_right > plate_left) and (ball_left < plate_right):
                # Successful bounce
                # Check on what block it bounced. The bounce angle should increase if it bounced on the outmost plates
                # and decrease on the inner plates

                # Change the angle
                angle_increase = False
                angle_decrease = False
                quarter_length = (plate_right-plate_left)/4
                ball_middle = (ball_left-plate_left) + self.ball_width/2
                if ball_middle < quarter_length:
                    # Most left
                    angle_increase = True
                elif ball_middle < 2*quarter_length:
                    # Left of middle
                    angle_decrease = True
                elif ball_middle < 3*quarter_length:
                    # Right of middle
                    angle_decrease = True
                elif ball_middle < 4*quarter_length:
                    # Most right
                    angle_increase = True

                if angle_increase:
                    sign = np.sign(self.angle)
                    self.angle += self.angle_increase*sign
                    if sign:
                        self.angle = np.maximum(self.angle_max, self.angle)
                    elif not sign:
                        self.angle = np.minimum(math.pi - self.angle_max, self.angle)

                elif angle_decrease:
                    sign = np.sign(-1*self.angle)
                    self.angle += self.angle_increase*sign
                    if not sign:
                        self.angle = np.maximum(self.angle_max, self.angle)
                    elif sign:
                        self.angle = np.minimum(math.pi - self.angle_max, self.angle)

                # Now that angle has been changed, update the new y position
                self.dy = -self.dy
                y_new = self.ball.y + self.dy * math.sin(self.angle)

                # Check for side bounce
                # There is a known bug here. If you move towards the ball it will get stuck in you
                ball_down = y_new + self.ball_height
                plate_up = player.plate_y_start
                right_bounce = (ball_left < plate_right) and (ball_right > plate_right) and (ball_down > plate_up)
                left_bounce = (ball_right > plate_left) and (ball_left < plate_left) and (ball_down > plate_up)
                if right_bounce or left_bounce:
                    self.dx = -self.dx
                    x_new = self.ball.x + self.dx * abs(math.cos(self.angle))

        # Check for collision with breakable_blocks
        # First make sure that the ball is high enough up to collide
        if y_new <= break_able_blocks.highest_y + break_able_blocks.block_height:
            collision = False
            blocks_to_remove = []

            for i in range(len(break_able_blocks.blocks)):
                block = break_able_blocks.blocks[i]
                # Check if right height first
                block_up = block.y
                block_down = block.y + block.height
                ball_up = y_new
                ball_down = y_new + self.ball_height
                if (ball_up <= block_down) and (ball_down >= block_up):
                    # Check if also x overlap
                    block_left = block.x
                    block_right = block.x + block.width
                    ball_left = x_new
                    ball_right = x_new + self.ball_width
                    if (ball_right >= block_left) and (ball_left <= block_right):
                        blocks_to_remove.append(block)
                        collision = True

                        # Check for side bounce
                        right_bounce = (ball_left < block_right) and (ball_right > block_right) and (ball_down > ball_up)
                        left_bounce = (ball_right > block_left) and (ball_left < block_left) and (ball_down > ball_up)
                        if right_bounce or left_bounce:
                            self.dx = -self.dx
                            x_new = self.ball.x + self.dx * abs(math.cos(self.angle))

            if collision:
                # Remove the blocks which collided
                for block in blocks_to_remove:
                    break_able_blocks.blocks.remove(block)

                # Update the lowest y at which there is a block
                break_able_blocks.update_highest_y()

                # Give the player scores
                player.increase_score(len(blocks_to_remove))

                self.dy = -self.dy
                y_new = self.ball.y + self.dy * math.sin(self.angle)



        self.ball.x = x_new
        self.ball.y = y_new

    def render(self, display):
        self.ball.render(display)
