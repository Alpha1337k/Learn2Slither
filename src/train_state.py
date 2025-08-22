from enum import Enum
from random import randint
from typing import List, Tuple


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
