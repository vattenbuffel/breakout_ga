import pygame


class Square:
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    def render(self, display):
        pygame.draw.rect(display, self.color, [self.x, self.y, self.width, self.height])


def squares_render(squares, display):
    for square in squares:
        square.render(display)
