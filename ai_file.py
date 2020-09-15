import numpy as np
import copy
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from player_file import Player
from ball_file import Ball
from break_able_blocks_file import Breakable_blocks
from game_file import Game
from multiprocessing import Pool


class AI:
    def __init__(self):
        self.d = 5
        self.n_var = 7
        self.n_genes = 100*self.n_var
        self.n_steps_look_into_future = 5

        self.mutate_probability = 1 / (self.n_var * self.n_genes)*10
        self.genes = []
        self.init_genes()
        self.vars = []
        self.function_score = 0
        self.fitness = 0

    def init_genes(self):
        self.genes = np.random.randint(2, size=self.n_var * self.n_genes)

    def mutate(self):
        for i in range(len(self.genes)):
            if np.random.rand() < self.mutate_probability:
                self.genes[i] = 1 - self.genes[i]

    # This plays a whole game
    def eval(self, game):
        n_sequences = self.play_one_game(game, kill_after_not_getting_points_for_n_rounds=200)

        self.function_score = game.player.score
        self.fitness = game.player.score / n_sequences

    # This plays a game and returns how many sequences it played
    def play_one_game(self, game, kill_after_not_getting_points_for_n_rounds=False):
        # Play until dead or game done or not gotten points for n rounds
        sequence_counter = 0
        last_point_round = 0
        cur_points = 0
        while (not game.player.dead) and (not (len(game.break_able_blocks.blocks) == 0)):
            self.play_one_step(game)
            sequence_counter += 1
            if kill_after_not_getting_points_for_n_rounds and (game.player.score == cur_points):
                if sequence_counter*self.n_steps_look_into_future - last_point_round >= kill_after_not_getting_points_for_n_rounds:
                    game.player.dead = True
            else:
                cur_points = game.player.score
                last_point_round = sequence_counter*self.n_steps_look_into_future

        return sequence_counter

    # Simulate all possible board states self.steps_ahead movements ahead. Calculate the score of those boards and pick
    # The best. Execute the best sequence of actions.
    def play_one_step(self, game):
        movement_sequences = self.generate_movement_sequences()

        # Save the game state
        saved_player_state = copy.deepcopy(game.player)
        saved_ball_state = copy.deepcopy(game.ball)
        saved_breakable_blocks_state = copy.deepcopy(game.break_able_blocks)

        # Play the game to find the best movement sequence
        best_score = -10 ** 10  # Hopefully this is low enough
        best_sequence = []
        for sequence in movement_sequences:
            self.execute_one_movement_sequence(sequence, game)
            cur_score = self.calc_score_of_game_state(game)

            if cur_score > best_score:
                best_score = cur_score
                best_sequence = sequence

            # Restore the game to the state it was before we started playing
            game.player = copy.deepcopy(saved_player_state)
            game.ball = copy.deepcopy(saved_ball_state)
            game.break_able_blocks = copy.deepcopy(saved_breakable_blocks_state)

        # This code will run the best sequence and show it, it won't affect the actual game state though
        """
        self.execute_one_movement_sequence(best_sequence, game, False)
        game.player = copy.deepcopy(saved_player_state)
        game.ball = copy.deepcopy(saved_ball_state)
        game.break_able_blocks = copy.deepcopy(saved_breakable_blocks_state)
        """
        # Take the best step
        self.execute_one_movement_sequence(best_sequence, game, simulate=False)

    # This executes a movement sequence, It requires a game state on which it can play
    def execute_one_movement_sequence(self, movement_sequence, game, simulate=True):
        for movement in movement_sequence:
            game.update(simulate, movement=movement)

    # This takes a game and calculates the score given by the ball, player and breakable blocks
    def calc_score_of_game_state(self, game):
        score = 0

        # Check if dead
        if game.player.dead:
            score += self.vars[0]

        # Check distance in x direction
        delta_x1 = abs(game.ball.ball.x - game.player.plate[0].x)
        delta_x2 = abs(game.player.plate[-1].x - game.ball.ball.x)
        score += delta_x1 * self.vars[1] + delta_x2 * self.vars[2]

        # Check speed
        speed = (game.ball.dx**2 + game.ball.dy**2)**0.5
        score += speed*self.vars[3]

        # Check angle
        score += abs(game.ball.angle)*self.vars[4]

        # Check y_pos of ball
        score += game.ball.ball.y * self.vars[5]

        # Check points of player
        score += game.player.score * self.vars[6]

        return score

    def generate_movement_sequences(self):
        string_movement_sequences = ["{0:b}".format(i) for i in range(2 ** self.n_steps_look_into_future)]

        # Add 0 to all the strings which are too short
        for i in range(len(string_movement_sequences)):
            sequence = string_movement_sequences[i]
            if len(sequence) < self.n_steps_look_into_future:
                n_too_short = self.n_steps_look_into_future - len(sequence)
                sequence = '0' * n_too_short + sequence
                string_movement_sequences[i] = sequence

        # Convert the string movement sequences into sequences containing the words "LEFT", "RIGHT"
        movement_sequences = []
        for string_sequence in string_movement_sequences:
            tmp = []
            for char in string_sequence:
                if char == '0':
                    tmp.append("LEFT")
                elif char == '1':
                    tmp.append("RIGHT")
                movement_sequences.append(tmp)
        return movement_sequences

    def decode(self):
        self.vars = []
        for i in range(self.n_var):
            cur_var = 0
            var_done = i
            for j in range(self.n_genes):
                cur_var = cur_var + 2 ** (-(j + 1)) * self.genes[var_done * self.n_genes + j]

            cur_var = -self.d + 2 * self.d / (1 - 2 ** (-self.n_genes)) * cur_var
            self.vars.append(cur_var)


def cross(chromosome1, chromosome2):
    n_genes = chromosome1.size
    cross_over_point = np.random.randint(n_genes)

    chromosome1_copy = np.copy(chromosome1)

    chromosome1[cross_over_point:n_genes] = chromosome2[cross_over_point:n_genes]
    chromosome2[cross_over_point:n_genes] = chromosome1_copy[cross_over_point:n_genes]


def create_game():
    display_width = 640
    display_height = 480
    game_display = pygame.display.set_mode((display_width, display_height))

    player = Player(display_width, display_height)
    ball = Ball(display_width, display_height)
    breakable_blocks = Breakable_blocks(display_width, display_height)
    game = Game(player, ball, breakable_blocks, display_width, display_height, game_display)

    return game


# This is for the multiprocessing to work
# Data here is a list of first the ai then the a game, ball and player state tuple
def dummy_eval(data):
    ai_temp = data[0]
    states = data[1]
    saved_player_state, saved_ball_state, saved_breakable_blocks_state = states

    game = create_game()
    game.player = saved_player_state
    game.break_able_blocks = saved_breakable_blocks_state
    game.ball = saved_ball_state
    ai_temp.eval(game)
    return ai_temp


class AI_population:
    def __init__(self, game):
        self.popSize = 50
        self.tournement_probability = 0.75
        self.tournement_size = 2
        self.best_individual_copies = 1
        self.n_generations = 25
        self.cross_probability = 0.8
        self.game = game

        self.population = [AI() for i in range(self.popSize)]
        assert (self.popSize % 2 == 0), "Invalid population size. It must be even."

    # Perform tournament_select for whole population
    def tournament_selection(self):
        parents = np.zeros(self.popSize)
        population_fitness = np.array([individual.fitness for individual in self.population])
        for i in range(self.popSize):
            parent = self.tournament_select(population_fitness)
            parents[i] = parent
        return parents

    # This picks a parent
    def tournament_select(self, population_fitness):
        # pick tournamentSize random individuals
        # Roll a random number r, if r <  probabilityHighestFitness pick the
        # highest fitness individual otherwhise remove that individual and keep
        # going until a indivual is picked or there is only 1 left

        # Make a unique list of contenders
        contender_indices = np.random.randint(self.popSize, size=self.tournement_size)
        while not (np.unique(contender_indices).size == self.tournement_size):
            contender_indices = np.random.randint(self.popSize, size=self.tournement_size)

        while 1:
            if np.random.rand() < self.tournement_probability:
                # Select the individual with highest fitness
                contender_fitness = population_fitness[contender_indices]
                winner_fitness = np.max(contender_fitness)
                return np.where(population_fitness == winner_fitness)[0][0]
            else:
                # Remove the contender with highest fitenss
                contender_fitness = population_fitness[contender_indices]
                index_highest_fitness = np.argmax(contender_fitness)
                contender_indices = np.delete(contender_indices, index_highest_fitness)

                # Check if only 1 contender left
                if contender_indices.size == 1:
                    return contender_indices[0]

    def mutate_population(self):
        for individual in self.population:
            individual.mutate()

    def decode_population(self):
        for individual in self.population:
            individual.decode()

    def eval_population(self):
        saved_player_state = copy.deepcopy(self.game.player)
        saved_ball_state = copy.deepcopy(self.game.ball)
        saved_breakable_blocks_state = copy.deepcopy(self.game.break_able_blocks)
        states = (saved_player_state, saved_ball_state, saved_breakable_blocks_state)

        pool = Pool(processes=8)
        ai_list = [[self.population[i], states] for i in range(self.popSize)]
        self.population = pool.map(dummy_eval, ai_list)
        pool.close()
        pool.join()

    def insert_best_individual(self, best_individual):
        for i in range(self.best_individual_copies):
            self.population[i] = best_individual

    def run_generations(self):
        for generation in range(self.n_generations):
            self.decode_population()
            self.eval_population()

            # Find best individual
            best_individual = self.population[0]
            for individual in self.population:
                if individual.fitness > best_individual.fitness:
                    best_individual = individual

            # Find parents
            parents_indices = self.tournament_selection().astype(np.int32)

            # Create new population
            new_population = []
            for i in range(self.popSize - 1):
                parent1Index = parents_indices[i]
                parent2Index = parents_indices[i + 1]
                parent1 = copy.deepcopy(self.population[parent1Index])
                parent2 = copy.deepcopy(self.population[parent2Index])

                new_population.append(parent1)
                new_population.append(parent2)
                # Should they cross?
                if np.random.rand() < self.cross_probability:
                    cross(parent1.genes, parent2.genes)

                i += 1

            self.population = new_population
            self.mutate_population()
            self.insert_best_individual(best_individual)
            print("Done with generation", generation, ".The best score achieved is:", best_individual.function_score, "with fitness", best_individual.fitness,
                  "and it's params is:", best_individual.vars)


def mutate_test():
    ai = AI()
    ai.n_genes = 10000
    mutate_probabilitites = [0, 0.1, 0.05, 0.07]
    epsilon = 0.01

    for probability in mutate_probabilitites:
        ai.genes = np.zeros(ai.n_var * ai.n_genes)
        ai.mutate_probability = probability
        ai.mutate()
        assert (np.abs(np.mean(ai.genes) - probability) < epsilon), "Mutate failed"


def decode_test():
    ai = AI()
    ai.genes = np.ones(ai.n_var * ai.n_genes)
    ai.decode()
    assert (ai.vars[0] == ai.vars[-1] == ai.d), "Decoding failed"
    ai.genes = np.zeros(ai.n_var * ai.n_genes)
    ai.decode()
    assert (ai.vars[0] == ai.vars[-1] == -ai.d), "Decoding failed"


def tournament_selection_test():
    ai_population = AI_population()
    ai_population.popSize = 5
    ai_population.tournement_size = 5
    fitness = np.array([1, 2, 3, 4, 5])

    ai_population.tournement_probability = 1
    assert (ai_population.tournament_select(fitness) == 4), "Tournament select failed"
    ai_population.tournement_probability = 0
    assert (ai_population.tournament_select(fitness) == 0), "Tournament select failed"


def cross_test():
    ai_pop = AI_population()
    ai1 = ai_pop.population[0]
    ai1.genes = np.zeros(ai1.n_var * ai1.n_genes)
    ai2 = ai_pop.population[1]
    ai2.genes = np.ones(ai2.n_var * ai2.n_genes)

    cross(ai1.genes, ai2.genes)
    sum_chromosome = np.sum(ai1.genes) + np.sum(ai2.genes)
    assert (sum_chromosome / ai1.genes.size == 1), "cross failed"


def best_individual_test():
    ai_pop = AI_population()
    best_individual = 0
    ai_pop.insert_best_individual(best_individual)
    for i in range(ai_pop.best_individual_copies):
        assert (ai_pop.population[i] == best_individual), "Insert_best_individual failed"


def myfunc(p):
    return p


if __name__ == '__main__':
    mutate_test()
    decode_test()
    # tournament_selection_test()
    # cross_test()
    # best_individual_test()

    game = create_game()

    ai_pop = AI_population(game)
    ai_pop.run_generations()
