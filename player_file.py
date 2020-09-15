import pygame
import square_file


class Player():
    def __init__(self, display_width, display_height):
        self.plate_length_n = 5
        self.plate_square_width = 25
        self.plate_square_height = 25
        self.score_per_block = 10
        self.score_factor = 2

        self.plate_y_offset = 50
        self.plate_color = (255, 0, 0)  # Red
        self.dead = False
        self.score = 0
        self.display_width = display_width
        self.display_height = display_height
        self.plate_x_start = display_width / 2 - self.plate_length_n / 2 * self.plate_square_width
        self.plate_y_start = display_height - self.plate_y_offset
        self.plate = [
            square_file.Square(self.plate_x_start + self.plate_square_width * i, self.plate_y_start, self.plate_square_width,
                               self.plate_square_height, self.plate_color) for i in range(self.plate_length_n)]

    def render(self, display):
        square_file.squares_render(self.plate, display)

    def increase_score(self, block_destroyed):
        self.score += self.score_per_block*self.score_factor**block_destroyed

    def movement(self, action):
        # Possible movements:
        # "RIGHT"
        # "LEFT"

        # Move right
        if action == "RIGHT":
            # Check for collision
            new_x = self.plate[-1].x + self.plate_square_width*2
            if new_x >= self.display_width:
                # If it will move out of bounds move it as far as it can go
                dx = self.display_width - (self.plate[-1].x + self.plate_square_width)
                for square in self.plate:
                    square.x += dx
                return
            else:
                for square in self.plate:
                    square.x += self.plate_square_width
                return

        # Move left
        elif action == "LEFT":
            # Check for collision
            new_x = self.plate[0].x - self.plate_square_width
            if new_x <= 0:
                # If it will move out of bounds move it as far as it can go
                dx = self.plate[0].x
                for square in self.plate:
                    square.x -= dx
                return
            else:
                for square in self.plate:
                    square.x -= self.plate_square_width
                return


