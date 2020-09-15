import pygame


class Game():
    def __init__(self, player, ball, break_able_blocks, display_width, display_height, game_display):
        self.fps = 30

        self.display_width = display_width
        self.display_height = display_height
        self.game_display = game_display
        pygame.display.set_caption('Game')
        self.clock = pygame.time.Clock()
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)
        self.player = player
        self.ball = ball
        self.break_able_blocks = break_able_blocks

    """
    If simulate is true then it doesn't render or care about frames per second. 
    This is to be able to look into the future.
    If the computer plays then movement is what should happen
    """
    def update(self, simulate, movement=None):
        self.handle_inputs()

        # If the player is dead allow no more inputs
        if not self.player.dead:
            if movement is not None:
                self.player.movement(movement)

            self.ball.update(self.player, self.break_able_blocks)

            if not simulate:
                self.game_display.fill(self.white)
                self.player.render(self.game_display)
                self.break_able_blocks.render(self.game_display)
                self.ball.render(self.game_display)
                pygame.display.update()
                self.clock.tick(self.fps)

            if self.ball.out_of_bound:
                self.player.dead = True

    def handle_inputs(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:  # Not checking for KEYUP
                if event.key == pygame.K_ESCAPE:
                    quit()
                elif event.key == pygame.K_a:
                    self.player.movement("LEFT")
                elif event.key == pygame.K_d:
                    self.player.movement("RIGHT")
