from square_file import Square
from square_file import squares_render


class Breakable_blocks():
    def __init__(self, display_width, display_height):
        self.block_width_n = 10
        self.block_height = 20
        self.block_height_n = 5

        self.color = (255, 140, 0)  # darkorange
        self.display_width = display_width
        self.display_height = display_height
        self.block_width = self.display_width / self.block_width_n

        self.blocks = [
            Square(x * self.block_width, y * self.block_height, self.block_width, self.block_height, self.color) for x in range(self.block_width_n) for y in range(self.block_height_n)]

        self.highest_y = 0
        self.update_highest_y()

    def render(self, display):
        squares_render(self.blocks, display)

    def update_highest_y(self):
        for block in self.blocks:
            if block.y > self.highest_y:
                self.highest_y = block.y