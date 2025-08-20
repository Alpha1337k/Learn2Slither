from email.policy import default
from operator import index
from random import randint, random
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

    def __init__(self):
        self.board = [[BoardPiece.EMPTY] * 10] * 10
        self.snake_body = []
        self.red_apples = []
        self.green_apples = []

        self.snake_length = 3
        self._place_snake()

    def _place_snake(self):
        pos_x = randint(0, 9)
        pos_y = randint(0, 9)

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
        self.snake_body.pop(-1)
        pos_x, pos_y = self.snake_body[0]

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

        pos_x, pos_y = self.snake_body[0]

        # if (pos_x, pos_y) in self.green_apples:
        #     self.green_apples.remove((pos_x, pos_y))
        #     return BoardPiece.GREEN

        # if (pos_x, pos_y) in self.red_apples:
        #     self.red_apples.remove((pos_x, pos_y))
        #     return BoardPiece.RED

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
        return -1

    data = {
        "wallLeftDist": min(get_distance(vision, -1, False, BoardPiece.WALL), 4),
        "wallRightDist": min(get_distance(vision, 1, False, BoardPiece.WALL), 4),
        "wallUpDist": min(get_distance(vision, -1, True, BoardPiece.WALL), 4),
        "wallDownDist": min(get_distance(vision, 1, True, BoardPiece.WALL), 4),
        # "greenLeftDist": get_distance(vision, -1, False, BoardPiece.GREEN),
        # "greenRightDist": get_distance(vision, 1, False, BoardPiece.GREEN),
        # "greenUpDist": get_distance(vision, -1, True, BoardPiece.GREEN),
        # "greenDownDist": get_distance(vision, 1, True, BoardPiece.GREEN),
        # "redLeftDist": get_distance(vision, -1, False, BoardPiece.RED),
        # "redRightDist": get_distance(vision, 1, False, BoardPiece.RED),
        # "redUpDist": get_distance(vision, -1, True, BoardPiece.RED),
        # "redDownDist": get_distance(vision, 1, True, BoardPiece.RED),
        # "snakeLeftDist": get_distance(vision, -1, False, BoardPiece.SNAKE),
        # "snakeRightDist": get_distance(vision, 1, False, BoardPiece.SNAKE),
        # "snakeUpDist": get_distance(vision, -1, True, BoardPiece.SNAKE),
        # "snakeDownDist": get_distance(vision, 1, True, BoardPiece.SNAKE),
    }

    return data


@validate_call(config=model_config)
def train_model(epochs: int, visual: bool):
    Q_table = {}
    learning_rate = 0.8
    discount_factor = 0.95
    exploration_prob = 0.2

    # state = TrainState()

    # print(state.snake_body[0])

    # print_snake_vision(state.get_snake_vision(), state.snake_body[0])

    # print(compress_vision(state.get_snake_vision(), state.snake_body[0]))
    np.set_printoptions(linewidth=np.inf)

    for epoch in range(epochs):
        print("--------")
        state = TrainState()

        need_stop = False
        step = 0

        while need_stop == False:
            current_values = compress_vision(
                state.get_snake_vision(), state.snake_body[0]
            )

            print(print_snake_vision(state.get_snake_vision(), state.snake_body[0]))

            current_state = list(current_values.values())

            if tuple(current_state) not in Q_table:
                Q_table[tuple(current_state)] = np.zeros(4)

            print(current_values, Q_table[tuple(current_state)])

            if np.random.rand() < exploration_prob:
                action = randint(0, 3)
                print("Random!")
            else:
                action = int(np.argmax(Q_table[tuple(current_state)]))

            result = state.move_snake(Direction(action))
            new_state = list(
                compress_vision(state.get_snake_vision(), state.snake_body[0]).values()
            )

            if tuple(new_state) not in Q_table:
                Q_table[tuple(new_state)] = np.zeros(4)

            reward = 0

            match result:
                case BoardPiece.WALL:
                    reward = -10
                    need_stop = True
                # case BoardPiece.SNAKE:
                #     reward = -10
                #     need_stop = True
                # case BoardPiece.GREEN:
                #     reward = 1
                #     state.snake_length += 1
                #     state.snake_body.append(state.snake_body[-1])
                # case BoardPiece.RED:
                #     reward = -0.5
                #     state.snake_length -= 1
                #     state.snake_body.pop(-1)
                #     if len(state.snake_body) == 0:
                #         need_stop = True
                #         reward = -10
                case BoardPiece.EMPTY:
                    reward = 0.1

            score = (1 - learning_rate) * Q_table[tuple(current_state)][
                action
            ] + learning_rate * (
                reward + discount_factor * np.max(Q_table[tuple(new_state)])
            )

            print(step, len(state.snake_body), Direction(action), need_stop)

            Q_table[tuple(current_state)][action] = score

            step += 1
