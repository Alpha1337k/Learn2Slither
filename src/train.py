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
    N = 1
    E = 2
    S = 3
    W = 4

# class syntax
class BoardPiece(Enum):
    EMPTY = 1
    GREEN = 2
    RED = 3
    SNAKE = 4
    HEAD = 5
    WALL = 6

def print_snake_vision(vision: List[List[BoardPiece]], head: Tuple[int, int]):
    print(vision)

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
            match randint(0, 3):
                case Direction.N:
                    if pos_y - 1 < 0:
                        continue
                    self.snake_body.append((pos_x, pos_y - 1))
                case Direction.E:
                    if pos_x + 1 > 9:
                        continue
                    self.snake_body.append((pos_x + 1, pos_y))
                case 2:
                    if pos_y + 1 > 9:
                        continue
                    self.snake_body.append((pos_x, pos_y + 1))
                case 3:
                    if pos_x - 1 < 0:
                        continue
                    self.snake_body.append((pos_x - 1, pos_y))


            pos_x, pos_y = self.snake_body[-1]
            i += 1

    def get_snake_vision(self) -> List[List[int]]:
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

        return [[x.value for x in x_vision], [y.value for y in y_vision]]

    def move_snake(self, direction: Direction) -> BoardPiece:
        self.snake_body.pop(-1)
        pos_x, pos_y = self.snake_body[0]

        assert (0 <= pos_x <= 9)
        assert (0 <= pos_y <= 9)

        match direction:
            case Direction.N:
                if pos_y - 1 < 0:
                    return BoardPiece.WALL
                self.snake_body.insert(0, (pos_x, pos_y - 1))
            case Direction.E:
                if pos_x + 1 > 9:
                    return BoardPiece.WALL
                self.snake_body.insert(0, (pos_x + 1, pos_y))
            case 2:
                if pos_y + 1 > 9:
                    return BoardPiece.WALL
                self.snake_body.insert(0, (pos_x, pos_y + 1))
            case 3:
                if pos_x - 1 < 0:
                    return BoardPiece.WALL
                self.snake_body.insert(0, (pos_x - 1, pos_y))

        pos_x, pos_y = self.snake_body[0]

        if self.green_apples.index((pos_x, pos_y)):
            self.green_apples.remove((pos_x, pos_y))
            return BoardPiece.GREEN

        if self.red_apples.index((pos_x, pos_y)):
            self.red_apples.remove((pos_x, pos_y))
            return BoardPiece.RED

        return BoardPiece.EMPTY



@validate_call(config=model_config)
def train_model(epochs: int, visual: bool):
    state = TrainState()

    print(state.snake_body[0])

    print_snake_vision(state.get_snake_vision(), state.snake_body[0])

    return

    Q_table = np.zeros((2, 10, 4))
    learning_rate = 0.8
    discount_factor = 0.95
    exploration_prob = 0.2
    
    need_stop = False

    while need_stop == False:
        current_state = state.get_snake_vision()

        if np.random.rand() < exploration_prob:
            action = randint(0, 3)
            print("Random!")
        else:
            action = np.argmax(Q_table[current_state])
        
        result = state.move_snake(Direction(action))
        new_state = state.get_snake_vision()

        reward = 0

        match result:
            case BoardPiece.WALL:
                reward = -10
                need_stop = True
            case BoardPiece.SNAKE:
                reward = -10
                need_stop = True
            case BoardPiece.GREEN:
                reward = 1
                state.snake_length += 1
                state.snake_body.append(state.snake_body[-1])
            case BoardPiece.RED:
                reward = -0.5
                state.snake_length -= 1
                state.snake_body.pop(-1)
                if len(state.snake_body) == 0:
                    need_stop = True
                    reward = -10
            case BoardPiece.EMPTY:
                reward = 0


        Q_table[current_state, action] = (1 - learning_rate) * Q_table + learning_rate * ( reward + discount_factor * max(Q_table[new_state, action]) )

    print(state.board)

