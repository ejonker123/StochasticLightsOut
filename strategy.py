"""
/*
 * Copyright (c) 2025 Jérôme Welscher
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
"""

import random
import csv
import os
import numpy as np
from ained import AiNed
from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    This Strategy interface declares operations common to all supported strategy versions.
    Concrete Strategies inherit from this interface.

    The Solver then uses this interface to call the algorithm defined by Concrete Strategies.

    Each strategy is supposed to calculate which cell to flip and also flip it. After flipping a cell, it should return a boolean.
    The purpose of the boolean is for debugging purposes if any logging is to be implemented or to keep track whether a step was taken or not.
    """

    def __init__(self, ained: AiNed):
        self.ained = ained

    @abstractmethod
    def solve(self, N: int, pos: tuple[int, int]) -> bool:
        pass

    @abstractmethod
    def __str__(self):
        pass


class StochasticStrategy_0(Strategy):
    """
    A very simple stochastic strategy that simply chooses a random light to flip.
    """

    def __init__(self, ained: AiNed):
        super().__init__(ained)

    def solve(self, N: int, pos: (int, int)) -> (int, int):
        board = self.ained.get_board(pos[0], pos[1], N, N)
        probs = []

        for i in range(len(board)):
            current_row = i // N
            current_col = i % N
            probs.append((current_row, current_col))

        row, column = probs[random.randint(0, len(probs) - 1)]
        self.ained.flip_lights(pos[0], pos[1], N, N, row, column)
        return True

    def __str__(self):
        return "Stochastic_0"


class StochasticStrategy_1(Strategy):
    """
    A very simple stochastic strategy that simply chooses a random light to flip.
    """

    def __init__(self, ained: AiNed):
        super().__init__(ained)

    def solve(self, N: int, pos: (int, int)) -> (int, int):
        board = self.ained.get_board(pos[0], pos[1], N, N)
        probs = []
        factor = 1  ## amount of times neighbour affect probability

        for i in range(len(board)):
            current_row = i // N
            current_col = i % N
            for x in range(-N - 1, N + 1):
                if x == 0:
                    probs.append(((i + x) // N, (i + x) % N))
                elif i + x >= 0 and i + x < len(board):
                    temp = self.ained.get_bit((i + x) // N, (i + x) % N)
                    if temp == 1:
                        for y in range(factor):
                            probs.append((current_row, current_col))

        row, column = probs[random.randint(0, len(probs) - 1)]
        self.ained.flip_lights(pos[0], pos[1], N, N, row, column)
        return True

    def __str__(self):
        return "Stochastic_1"


class StochasticStrategy_2(Strategy):
    """
    A very simple stochastic strategy that simply chooses a random light to flip.
    """

    def __init__(self, ained: AiNed):
        super().__init__(ained)

    def solve(self, N: int, pos: (int, int)) -> (int, int):
        board = self.ained.get_board(pos[0], pos[1], N, N)
        probs = []
        factor = 2  ## amount of times neighbour affect probability

        for i in range(len(board)):
            current_row = i // N
            current_col = i % N
            for x in range(-N - 1, N + 1):
                if x == 0:
                    probs.append(((i + x) // N, (i + x) % N))
                elif i + x >= 0 and i + x < len(board):
                    temp = self.ained.get_bit((i + x) // N, (i + x) % N)
                    if temp == 1:
                        for y in range(factor):
                            probs.append((current_row, current_col))

        row, column = probs[random.randint(0, len(probs) - 1)]
        self.ained.flip_lights(pos[0], pos[1], N, N, row, column)
        return True

    def __str__(self):
        return "Stochastic_2"


class StochasticGreed(Strategy):
    """
    A very simple stochastic strategy that simply chooses a random light to flip.
    """

    def __init__(self, ained: AiNed):
        super().__init__(ained)

    def solve(self, N: int, pos: (int, int)) -> (int, int):
        board = self.ained.get_board(pos[0], pos[1], N, N)
        probs = []
        factor = 0  ## amount of times neighbour affect probability

        for i in range(len(board)):
            current_row = i // N
            current_col = i % N
            if self.ained.get_bit(current_row, current_col) == 1:
                probs.append((current_row, current_col))

        row, column = probs[random.randint(0, len(probs) - 1)]
        self.ained.flip_lights(pos[0], pos[1], N, N, row, column)
        return True

    def __str__(self):
        return "StochasticGreed"


class StochasticLine(Strategy):
    def __init__(self, ained: AiNed):
        super().__init__(ained)

    def solve(self, N: int, pos: (int, int)) -> (int, int):
        board = self.ained.get_board(pos[0], pos[1], N, N)
        on = False
        for i in range(N):
            for j in range(N):
                if board[i * N + j] == 1:
                    row = i
                    column = random.randint(0, N - 1)
                    self.ained.flip_lights(pos[0], pos[1], N, N, row, column)
                    return True
        return False

    def __str__(self):
        return "StochasticLine"


class GreedySquare(Strategy):
    def __init__(self, ained: AiNed):
        super().__init__(ained)

    def solve(self, N: int, pos: (int, int)) -> (int, int):
        board = self.ained.get_board(pos[0], pos[1], N, N)
        rows = [0, N - 1]
        columns = [0, N - 1]
        size = N

        # [topleft,topmid, topright,midleft, midmid, midright, botleft, botmid, botright]

        while size > 3:
            size = size - 2
            counts = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(columns[0], columns[1] + 1):
                for j in range(rows[0], rows[1] + 1):
                    if i >= columns[0] and i <= columns[1] - 2:
                        if j >= rows[0] and j <= rows[1] - 2:
                            if board[i * N + j] == 1:
                                counts[0] += 1
                        if j >= rows[0] + 1 and j <= rows[1] - 1:
                            if board[i * N + j] == 1:
                                counts[1] += 1
                        if j >= rows[0] + 2 and j <= rows[1]:
                            if board[i * N + j] == 1:
                                counts[2] += 1
                    if i >= columns[0] + 1 and i <= columns[1] - 1:
                        if j >= rows[0] and j <= rows[1] - 2:
                            if board[i * N + j] == 1:
                                counts[3] += 1
                        if j >= rows[0] + 1 and j <= rows[1] - 1:
                            if board[i * N + j] == 1:
                                counts[4] += 1
                        if j >= rows[0] + 2 and j <= rows[1]:
                            if board[i * N + j] == 1:
                                counts[5] += 1
                    if i >= columns[0] + 1 and i <= columns[1]:
                        if j >= rows[0] and j <= rows[1] - 2:
                            if board[i * N + j] == 1:
                                counts[6] += 1
                        if j >= rows[0] + 1 and j <= rows[1] - 1:
                            if board[i * N + j] == 1:
                                counts[7] += 1
                        if j >= rows[0] + 2 and j <= rows[1]:
                            if board[i * N + j] == 1:
                                counts[8] += 1

            temp = max(counts)
            z = 0
            for i in range(len(counts)):
                if counts[i] == temp:
                    z += 1
            if z > 1:
                total = 0
                index = counts.index(temp)
                for j in range(len(counts)):
                    if counts[j] == temp:
                        if j == 0:
                            if total < counts[j + 1]:
                                total = counts[j + 1]
                                index = j
                        if j == len(counts) - 1:
                            if total < counts[j - 1]:
                                total = counts[j - 1]
                                index = j
                        if j != 0 and j != len(counts) - 1:
                            if total < (counts[j - 1] + counts[j + 1]):
                                total = counts[j - 1] + counts[j + 1]
                                index = j
                square = index
            else:
                square = counts.index(temp)

            if square == 0:
                rows[0] = rows[0]
                rows[1] = rows[1] - 2
                columns[0] = columns[0]
                columns[1] = columns[1] - 2
            if square == 1:
                rows[0] = rows[0] + 1
                rows[1] = rows[1] - 1
                columns[0] = columns[0]
                columns[1] = columns[1] - 2
            if square == 2:
                rows[0] = rows[0] + 2
                rows[1] = rows[1]
                columns[0] = columns[0]
                columns[1] = columns[1] - 2
            if square == 3:
                rows[0] = rows[0]
                rows[1] = rows[1] - 2
                columns[0] = columns[0] + 1
                columns[1] = columns[1] - 1
            if square == 4:
                rows[0] = rows[0] + 1
                rows[1] = rows[1] - 1
                columns[0] = columns[0] + 1
                columns[1] = columns[1] - 1
            if square == 5:
                rows[0] = rows[0] + 2
                rows[1] = rows[1]
                columns[0] = columns[0] + 1
                columns[1] = columns[1] - 1
            if square == 6:
                rows[0] = rows[0]
                rows[1] = rows[1] - 2
                columns[0] = columns[0] + 2
                columns[1] = columns[1]
            if square == 7:
                rows[0] = rows[0] + 1
                rows[1] = rows[1] - 1
                columns[0] = columns[0] + 2
                columns[1] = columns[1]
            if square == 8:
                rows[0] = rows[0] + 2
                rows[1] = rows[1]
                columns[0] = columns[0] + 2
                columns[1] = columns[1]
                # [stairs_tl, door_top, stairs_tr, door_left, full, door_right, stairs_bl, door_bot, stairs_br]
        shape_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        top_left = (columns[0]) * (N) + rows[0]
        top_mid = (columns[0]) * (N) + rows[0] + 1
        top_right = (columns[0]) * (N) + rows[0] + 2
        mid_left = (columns[0] + 1) * (N) + rows[0]
        mid_mid = (columns[0] + 1) * (N) + rows[0] + 1
        mid_right = (columns[0] + 1) * (N) + rows[0] + 2
        bot_left = (columns[0] + 2) * (N) + rows[0]
        bot_mid = (columns[0] + 2) * (N) + rows[0] + 1
        bot_right = (columns[0] + 2) * (N) + rows[0] + 2
        result = [top_left, top_mid, top_right, mid_left, mid_mid, mid_right, bot_left, bot_mid, bot_right]
        # check shape 1
        shape_0 = [0]
        shape_1 = [0]
        shape_2 = [0]
        shape_3 = [0]
        shape_4 = [0]
        shape_5 = [0]
        shape_6 = [0]
        shape_7 = [0]
        shape_8 = [0]

        if board[top_left] != 0:
            shape_0 = [board[top_left], board[top_mid], board[top_right], board[mid_left], board[mid_mid],
                       board[bot_left]]

        if board[top_mid] != 0:
            shape_1 = [board[top_left], board[top_mid], board[top_right], board[mid_left], board[mid_mid],
                       board[mid_right]]

        if board[top_right] != 0:
            shape_2 = [board[top_left], board[top_mid], board[top_right], board[mid_mid], board[mid_right],
                       board[bot_right]]

        if board[mid_left] != 0:
            shape_3 = [board[top_left], board[top_mid], board[mid_left], board[mid_mid], board[bot_left],
                       board[bot_mid]]

        if board[mid_mid] != 0:
            shape_4 = [board[top_left], board[top_mid], board[top_right], board[mid_left], board[mid_mid],
                       board[mid_right], board[bot_left], board[bot_mid], board[bot_right]]

        if board[mid_right] != 0:
            shape_5 = [board[top_mid], board[top_right], board[mid_mid], board[mid_right], board[bot_mid],
                       board[bot_right]]

        if board[bot_left] != 0:
            shape_6 = [board[top_left], board[mid_left], board[mid_mid], board[bot_left], board[bot_mid],
                       board[bot_right]]

        if board[bot_mid] != 0:
            shape_7 = [board[mid_left], board[mid_mid], board[mid_right], board[bot_left], board[bot_mid],
                       board[bot_right]]

        if board[bot_right] != 0:
            shape_8 = [board[top_right], board[mid_mid], board[mid_right], board[bot_left], board[bot_mid],
                       board[bot_right]]

        temp = [sum(shape_0), sum(shape_1), sum(shape_2), sum(shape_3), sum(shape_4), sum(shape_5), sum(shape_6),
                sum(shape_7), sum(shape_8)]
        max_1 = max(temp)
        best_shape = temp.index(max_1)

        row = result[best_shape] // N
        column = result[best_shape] % N
        self.ained.flip_lights(pos[0], pos[1], N, N, row, column)
        return True

    def __str__(self):
        return "greedySquare"


class GreedyStrategy(Strategy):
    """
    A greedy strategy that chooses the light that is the most likely to turn off as many lights as possible while minimising the amount of lights that turn on.
    """

    def __init__(self, ained: AiNed):
        super().__init__(ained)

    def solve(self, N: int, pos: tuple[int, int]) -> bool:
        min_delta_coords = (0, 0)
        min_delta = float("inf")
        current_board = self.ained.get_board(pos[0], pos[1], N, N)
        current_coefficients = self.ained.get_coefficients()
        for i in range(len(current_board)):
            current_row = i // N
            current_col = i % N
            temp_delta = 0
            for j in range(len(current_board)):
                if j == i:
                    if current_board[i] == 0:
                        temp_delta += self.greedy_eval(current_board[i], current_coefficients[0])
                    continue
                respective_row = j // N
                respective_col = j % N

                row_diff = abs(respective_row - current_row)
                col_diff = abs(respective_col - current_col)
                if row_diff < 5 and col_diff < 5:
                    temp_delta += self.greedy_eval(current_board[j], current_coefficients[row_diff * 5 + col_diff])
            if temp_delta < min_delta:
                min_delta_coords = (current_row, current_col)
                min_delta = min(min_delta, temp_delta)
        self.ained.flip_lights(pos[0], pos[1], N, N, min_delta_coords[0], min_delta_coords[1])
        return True

    def greedy_eval(self, light_value: int, probability: float) -> float:
        """ Greedy formula """
        return probability * (1 - 2 * light_value)

    def __str__(self):
        return "Greedy"


class SimAnnStrategy(Strategy):
    """ A simulated annealing strategy that is based on MCMC with a decreasing temperature. """

    def __init__(self, ained: AiNed):
        super().__init__(ained)
        self.ained = ained
        os.makedirs("data/SimAnnRuns", exist_ok=True)
        columns = ["curr_step", "curr_energy", "proposed_step", "proposed_energy", "accepted_board", "accepted_bool"]

        # Name file
        self.i = 1
        self.file_name = "SimAnn_simulation_"
        while os.path.exists(f"data/SimAnnRuns/{self.file_name}{self.i}.csv"):
            self.i += 1

        # Add columns if file is new
        with open(f"data/SimAnnRuns/{self.file_name}{self.i}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)

        self.curr_step = 1
        self.start_T = 10.0
        self.min_T = 1
        self.T = self.start_T
        self.cool = 0.95
        self.acceptance_window = 1000
        self.recent_accepts = list()

    def solve(self, N: int, pos: tuple[int, int]) -> bool:
        board = self.ained.get_board(pos[0], pos[1], N, N)
        curr_energy = energy(board)
        row, col = np.random.randint(0, N, size=2)
        proposed_energy = self.estimate_energy(board, N, row, col)
        delta_energy = proposed_energy - curr_energy
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / self.T):
            self.ained.flip_lights(pos[0], pos[1], N, N, row, col)
            newer_board = self.ained.get_board(pos[0], pos[0], N, N)
            self.save_step(curr_energy, (row, col), proposed_energy, newer_board, accepted=True)
            self.cool_down()
            self.show_temp()
            return True
        else:
            self.save_step(curr_energy, (row, col), proposed_energy, board, accepted=False)
            self.cool_down()
            return False

    def save_step(self, curr_energy: int, proposed_step: tuple[int, int], new_energy: int, accepted_board: list[int],
                  accepted: bool):
        """
        Special function that logs and keeps track of energy and board states throughout Simulated Annealing.
        It also keeps track of the acceptance ratio via the recently accepted steps.
        """

        self.recent_accepts.append(1 if accepted else 0)
        if len(self.recent_accepts) > self.acceptance_window:
            self.recent_accepts.pop(0)

        new_row = [self.curr_step, curr_energy, proposed_step, new_energy, accepted_board, accepted]
        with open(f"data/SimAnnRuns/{self.file_name}{self.i}.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(new_row)
        if accepted:
            self.curr_step += 1

    def __str__(self):
        return "SimAnn"

    def cool_down(self):
        self.T *= self.cool
        self.T = max(self.min_T, self.T)

    def show_temp(self):
        """ Print the current temperature and acceptance ratio """
        A = sum(self.recent_accepts) / len(self.recent_accepts)
        print(f"Current Accept Rate: {A:0.2f}, Current Heat: {self.T:0.2f}")

    def estimate_energy(self, board: list[int], N: int, row: int, col: int, num_estimations=10):
        """ Estimate the likely energy of a given board state after flipping a light """
        estimated_energies = list()
        for i in range(num_estimations):
            coeff = self.ained.get_coefficients()
            for index in range(len(board)):
                respective_row = index // N
                respective_col = index % N

                row_diff = abs(respective_row - row)
                col_diff = abs(respective_col - col)

                if row_diff < 5 and col_diff < 5:
                    curr_coeff = coeff[row_diff * 5 + col_diff]
                    if random.random() < curr_coeff:
                        board[index] = 1 - board[index]
            estimated_energies.append(energy(board))

        return np.mean(estimated_energies)


def energy(board: list) -> int:
    sum_energy = 0
    for i in range(len(board)):
        sum_energy += int(board[i])
    return sum_energy
