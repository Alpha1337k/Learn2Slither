from email.policy import default
from itertools import count
from operator import index
from random import randint, random
from re import A
from typing import Dict, List, Tuple
from typing_extensions import Self
from pydantic import ConfigDict, validate_call
from dataclasses import dataclass
from enum import Enum
import numpy as np


model_config = ConfigDict(arbitrary_types_allowed=True)


class Direction(Enum):
    N = 0
    E = 1
    S = 2
    W = 3


# class syntax
class BoardPiece(Enum):
    EMPTY = 1
    GREEN = 2
    RED = 3
    SNAKE = 4
    HEAD = 5
    WALL = 6


def print_snake_vision(vision: List[List[BoardPiece]], head: Tuple[int, int]):
    def to_print(key):
        char = "??"
        match BoardPiece(key):
            case BoardPiece.EMPTY:
                char = "0"
            case BoardPiece.GREEN:
                char = "G"
            case BoardPiece.RED:
                char = "R"
            case BoardPiece.SNAKE:
                char = "S"
            case BoardPiece.HEAD:
                char = "H"
            case BoardPiece.WALL:
                char = "W"

        return char

    for y in range(12):

        if y == (head[1] + 1):
            for x in range(12):
                print(f"{to_print(vision[0][x])} ", end="")
            print("")
        else:
            print(f"{'  ' * (head[0] + 1)}{to_print(vision[1][y])}")


class TrainState:
    board: List[List[BoardPiece]]
    snake_length: int
    snake_body: List[Tuple[int, int]]
    red_apples: List[Tuple[int, int]]
    green_apples: List[Tuple[int, int]]

    def __place_apple(self, color: BoardPiece):
        while True:
            pos_x = randint(0, 9)
            pos_y = randint(0, 9)

            if (
                (pos_x, pos_y) not in self.snake_body
                and (pos_x, pos_y) not in self.red_apples
                and (pos_x, pos_y) not in self.green_apples
            ):
                if color == BoardPiece.RED:
                    self.red_apples.append((pos_x, pos_y))
                else:
                    self.green_apples.append((pos_x, pos_y))
                break

    def __init__(self):
        self.board = [[BoardPiece.EMPTY] * 10] * 10
        self.snake_body = []
        self.red_apples = []
        self.green_apples = []

        for _ in range(2):
            self.__place_apple(BoardPiece.GREEN)
        for _ in range(1):
            self.__place_apple(BoardPiece.RED)

        self.snake_length = 3
        self._place_snake()

    def _place_snake(self):
        pos_x = randint(3, 6)
        pos_y = randint(3, 6)

        self.snake_body.append((pos_x, pos_y))
        i = 0
        while i < 2:
            value = Direction(randint(0, 3))

            match value:
                case Direction.N:
                    if pos_y - 1 < 0:
                        continue
                    self.snake_body.append((pos_x, pos_y - 1))
                    i += 1
                case Direction.E:
                    if pos_x + 1 > 9:
                        continue
                    self.snake_body.append((pos_x + 1, pos_y))
                    i += 1
                case Direction.S:
                    if pos_y + 1 > 9:
                        continue
                    self.snake_body.append((pos_x, pos_y + 1))
                    i += 1
                case Direction.W:
                    if pos_x - 1 < 0:
                        continue
                    self.snake_body.append((pos_x - 1, pos_y))
                    i += 1

            pos_x, pos_y = self.snake_body[-1]

        assert len(self.snake_body) == 3

    def get_snake_vision(self) -> List[List[BoardPiece]]:
        pos_x, pos_y = self.snake_body[0]

        x_vision = [BoardPiece.WALL]
        for i in range(0, 10):
            if (i, pos_y) in self.green_apples != -1:
                x_vision.append(BoardPiece.GREEN)
            elif (i, pos_y) in self.red_apples != -1:
                x_vision.append(BoardPiece.RED)
            elif (pos_x, pos_y) == (i, pos_y):
                x_vision.append(BoardPiece.HEAD)
            elif (i, pos_y) in self.snake_body != -1:
                x_vision.append(BoardPiece.SNAKE)
            else:
                x_vision.append(BoardPiece.EMPTY)
        x_vision.append(BoardPiece.WALL)

        y_vision = [BoardPiece.WALL]
        for i in range(0, 10):
            if (pos_x, i) in self.green_apples != -1:
                y_vision.append(BoardPiece.GREEN)
            elif (pos_x, i) in self.red_apples != -1:
                y_vision.append(BoardPiece.RED)
            elif (pos_x, pos_y) == (pos_x, i):
                y_vision.append(BoardPiece.HEAD)
            elif (pos_x, i) in self.snake_body != -1:
                y_vision.append(BoardPiece.SNAKE)
            else:
                y_vision.append(BoardPiece.EMPTY)
        y_vision.append(BoardPiece.WALL)

        return [x_vision, y_vision]
        return [[x.value for x in x_vision], [y.value for y in y_vision]]

    def move_snake(self, direction: Direction) -> BoardPiece:
        assert len(self.snake_body) > 0

        pos_x, pos_y = self.snake_body[0]
        self.snake_body.pop(-1)

        assert 0 <= pos_x <= 9
        assert 0 <= pos_y <= 9

        match direction:
            case Direction.N:
                if pos_y - 1 < 0:
                    return BoardPiece.WALL
                self.snake_body.insert(0, (pos_x, pos_y - 1))
            case Direction.E:
                if pos_x + 1 > 9:
                    return BoardPiece.WALL
                self.snake_body.insert(0, (pos_x + 1, pos_y))
            case Direction.S:
                if pos_y + 1 > 9:
                    return BoardPiece.WALL
                self.snake_body.insert(0, (pos_x, pos_y + 1))
            case Direction.W:
                if pos_x - 1 < 0:
                    return BoardPiece.WALL
                self.snake_body.insert(0, (pos_x - 1, pos_y))
            case _:
                raise ValueError("Invalid direction")

        assert len(self.snake_body) > 0

        pos_x, pos_y = self.snake_body[0]

        if (pos_x, pos_y) in self.green_apples:
            self.green_apples.remove((pos_x, pos_y))
            self.__place_apple(BoardPiece.GREEN)
            return BoardPiece.GREEN

        if (pos_x, pos_y) in self.red_apples:
            self.red_apples.remove((pos_x, pos_y))
            self.__place_apple(BoardPiece.RED)
            return BoardPiece.RED

        if (pos_x, pos_y) in self.snake_body[1:]:
            return BoardPiece.SNAKE

        return BoardPiece.EMPTY


def compress_vision(vision: List[List[BoardPiece]], pos: Tuple[int, int]):
    def get_distance(
        vision: List[List[BoardPiece]], direction: int, vertical: bool, item: BoardPiece
    ) -> int:
        distance = 0
        arr_to_check = vision[0] if not vertical else vision[1]
        start_pos = pos[0] + 1 if not vertical else pos[1] + 1
        # direction: 1 for up, -1 for down
        delta = direction
        while 0 <= start_pos + delta * distance < len(vision[1]):
            check_val = start_pos + delta * distance
            if arr_to_check[check_val] == item:
                return distance
            distance += 1
        return 0

    compression = 2

    # states:
    # deathFar = 2
    # deathClose = 1
    # green = 0

    directions = [
        (-1, True),  # up
        (1, False),  # right
        (1, True),  # down
        (-1, False),  # left
    ]

    data = []

    for (direction, is_vertical) in directions:
        dir_score = min(
                get_distance(vision, direction, is_vertical, BoardPiece.WALL) or compression,
                get_distance(vision, direction, is_vertical, BoardPiece.SNAKE) or compression,
                get_distance(vision, direction, is_vertical, BoardPiece.RED) or compression,
            )
        if dir_score > 1 and get_distance(vision, direction, is_vertical, BoardPiece.GREEN) > 0:
            dir_score = 0
        data.append(dir_score)

    return tuple(data)


class QTable:
    def __init__(self, len=3):
        self.Q_Table = {}
        self.exploration_prob = 0.2
        self.learning_rate = 0.2
        self.discount_factor = 0.85

    def __rotate_state(self, state: Tuple):
        return state
        sort_count = []

        for i in range(4):
            rotated = state[i:] + state[:i]
            prev = rotated[0]
            count = 0
            for j in range(1, len(rotated)):
                if rotated[j] >= prev:
                    count += 1
                    prev = rotated[j]
                else:
                    break
            sort_count.append((count, i, rotated))

        best_rotation = max(sort_count, key=lambda x: x[0])

        return tuple(best_rotation[2])

    def get_weights(self, state: Tuple[int, int]) -> List[float]:
        rotated_state = self.__rotate_state(state)
        if rotated_state not in self.Q_Table:
            self.Q_Table[rotated_state] = np.zeros(4)

        return self.Q_Table[rotated_state]

    def get_action(self, state: Tuple) -> int:
        weights = self.get_weights(state)
        sorted_options = np.argsort(weights)

        if np.random.rand() < self.exploration_prob:
            action = sorted_options[randint(0, 3)]
            print("Random!")
        else:
            action = int(np.argmax(weights))

        print(f"Action: {state} = {weights} = {Direction(action)}")

        return action

    def set_reward(
        self,
        state: Tuple,
        new_state: Tuple | None,
        action: int,
        reward: float,
        stopped: bool,
    ):
        rotated_state = self.__rotate_state(state)
        if rotated_state not in self.Q_Table:
            self.Q_Table[rotated_state] = np.zeros(4)
            
        if stopped:
            score = (1 - self.learning_rate) * self.Q_Table[rotated_state][
                action
            ] + self.learning_rate * reward
        else:
            assert new_state is not None

            new_state_scores = self.get_weights(new_state)

            score = (1 - self.learning_rate) * self.Q_Table[rotated_state][
                action
            ] + self.learning_rate * (
                reward + self.discount_factor * np.max(new_state_scores)
            )

        self.Q_Table[rotated_state][action] = score


@validate_call(config=model_config)
def train_model(epochs: int, visual: bool):
    tables = QTable(3)
    learning_rate = 0.8
    discount_factor = 0.85
    exploration_prob = 0.2

    max_steps = 0
    max_length = 0

    # state = TrainState()

    # print(state.snake_body[0])

    # print_snake_vision(state.get_snake_vision(), state.snake_body[0])

    # print(compress_vision(state.get_snake_vision(), state.snake_body[0]))
    np.set_printoptions(linewidth=np.inf)

    max_len = 0
    max_steps = 0

    for epoch in range(epochs):
        print("--------")
        state = TrainState()

        need_stop = False
        step = 0

        while need_stop == False and step < 1000:
            assert len(state.green_apples) == 2
            assert len(state.red_apples) == 1

            current_state = compress_vision(
                state.get_snake_vision(), state.snake_body[0]
            )

            print_snake_vision(state.get_snake_vision(), state.snake_body[0])
            print(current_state)

            max_steps = max(max_steps, step)
            max_length = max(max_length, len(state.snake_body))

            action = tables.get_action(current_state)

            reward = 0
            result = state.move_snake(Direction(action))

            match result:
                case BoardPiece.WALL:
                    reward = -10
                    need_stop = True
                case BoardPiece.SNAKE:
                    reward = -10
                    need_stop = True
                case BoardPiece.GREEN:
                    reward = 12
                    state.snake_length += 1
                    state.snake_body.append(state.snake_body[-1])
                case BoardPiece.RED:
                    reward = -4
                    state.snake_length -= 1
                    state.snake_body.pop(-1)
                    if len(state.snake_body) == 0:
                        need_stop = True
                        reward = -10
                case BoardPiece.EMPTY:
                    reward = -0.2

            # if len(state.snake_body) == 10:
            #     need_stop = True
            #     reward = 10
            #     print("DING DING IDNG ")
            #     exit(1)

            if need_stop:
                tables.set_reward(current_state, None, action, reward, need_stop)
                break

            next_values = compress_vision(state.get_snake_vision(), state.snake_body[0])

            tables.set_reward(current_state, next_values, action, reward, need_stop)

            step += 1
    print(max_length, max_steps)

