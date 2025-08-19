from random import randint, random
from typing import List, Tuple
from typing_extensions import Self
from pydantic import ConfigDict, validate_call
from dataclasses import dataclass

model_config = ConfigDict(arbitrary_types_allowed=True)


from enum import Enum

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

class TrainState:
    board: List[List[BoardPiece]]
    snake_length: int
    snake_body: List[Tuple[int, int]]
    red_apples: List[Tuple[int, int]]
    green_apples: List[Tuple[int, int]]

    def __init__(self):
        self.board = [[BoardPiece.EMPTY] * 10] * 10
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


    def get_snake_vision(self) -> Tuple[List[str], List[str]]:
        pos_x, pos_y = self.snake_body[0]

        x_vision = ["W"]
        for i in range(0, 10):
            if self.green_apples.index((i, pos_y)) != -1:
                x_vision.append("G")
            elif self.red_apples.index((i, pos_y)) != -1:
                x_vision.append("G")
            elif (pos_x, pos_y) == (i, pos_y):
                x_vision.append("H")
            elif self.snake_body.index((i, pos_y)) != -1:
                x_vision.append("S")
            else:
                x_vision.append("0")
        x_vision.append("W")

        y_vision = ["W"]
        for i in range(0, 10):
            if self.green_apples.index((pos_x, i)) != -1:
                y_vision.append("G")
            elif self.red_apples.index((pos_x, i)) != -1:
                y_vision.append("G")
            elif (pos_x, pos_y) == (pos_x, i):
                y_vision.append("H")
            elif self.snake_body.index((pos_x, i)) != -1:
                y_vision.append("S")
            else:
                y_vision.append("0")
        y_vision.append("W")

        return (x_vision, y_vision)

    def _move_snake(self, direction: Direction) -> bool:
        self.snake_body.pop(-1)
        pos_x, pos_y = self.snake_body[0]

        assert (0 <= pos_x <= 9)
        assert (0 <= pos_y <= 9)

        match direction:
            case Direction.N:
                if pos_y - 1 < 0:
                    return False
                self.snake_body.insert(0, (pos_x, pos_y - 1))
            case Direction.E:
                if pos_x + 1 > 9:
                    return False
                self.snake_body.insert(0, (pos_x + 1, pos_y))
            case 2:
                if pos_y + 1 > 9:
                    return False
                self.snake_body.insert(0, (pos_x, pos_y + 1))
            case 3:
                if pos_x - 1 < 0:
                    return False
                self.snake_body.insert(0, (pos_x - 1, pos_y))
        return True



@validate_call(config=model_config)
def train_model(sessions: int, visual: bool):
    state = TrainState()

    print(state.board)

