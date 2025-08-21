from email.policy import default
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

    data = [
        (
            min(
                get_distance(vision, -1, True, BoardPiece.WALL),
                get_distance(vision, -1, True, BoardPiece.SNAKE),
                compression,
            ),  # deathUpDist
            min(
                get_distance(vision, 1, False, BoardPiece.WALL),
                get_distance(vision, 1, False, BoardPiece.SNAKE),
                compression,
            ),  # deathRightDist
            min(
                get_distance(vision, 1, True, BoardPiece.WALL),
                get_distance(vision, 1, True, BoardPiece.SNAKE),
                compression,
            ),  # deathDownDist
            min(
                get_distance(vision, -1, False, BoardPiece.WALL),
                get_distance(vision, -1, False, BoardPiece.SNAKE),
                compression,
            ),  # deathLeftDist
        ),
        (
            min(
                get_distance(vision, -1, True, BoardPiece.GREEN), compression
            ),  # greenUpDist
            min(
                get_distance(vision, 1, True, BoardPiece.GREEN), compression
            ),  # greenDownDist
            min(
                get_distance(vision, 1, False, BoardPiece.GREEN), compression
            ),  # greenRightDist
            min(
                get_distance(vision, -1, False, BoardPiece.GREEN), compression
            ),  # greenLeftDist
        ),
        (
            min(
                get_distance(vision, -1, True, BoardPiece.RED), compression
            ),  # redUpDist
            min(
                get_distance(vision, 1, False, BoardPiece.RED), compression
            ),  # redRightDist
            min(
                get_distance(vision, 1, True, BoardPiece.RED), compression
            ),  # redDownDist
            min(
                get_distance(vision, -1, False, BoardPiece.RED), compression
            ),  # redLeftDist
        ),
    ]

    data = [
        tuple(1 if x > 1 and i != 0 else x for x in group)
        for i, group in enumerate(data)
    ]

    return data


class QTable:
    def __init__(self, len=3):
        self.Q_Table = [{} for _ in range(len)]
        self.exploration_prob = 0.2
        self.learning_rate = 0.2
        self.discount_factor = 0.85

    def __get_action(self, table: Dict, state: Tuple):
        if state not in table:
            table[state] = np.zeros(4)

        action = -1

        sorted_options = np.argsort(table[state])

        if np.random.rand() < self.exploration_prob:
            action = sorted_options[randint(0, 2)]
            print("Random!")
        else:
            action = int(np.argmax(table[state]))

        return table[state]

    def get_weights(self, state: Tuple[int, int], table_idx: int) -> List[float]:
        if state not in self.Q_Table[table_idx]:
            self.Q_Table[table_idx][tuple(state)] = np.zeros(4)

        return self.Q_Table[table_idx][tuple(state)]

    def get_action(self, states: List[Tuple]) -> int:
        final_state = []

        for table, state in zip(self.Q_Table, states):
            final_state.append(self.__get_action(table, state))

        sum_state = np.sum(final_state, axis=0)

        sorted_options = np.argsort(sum_state)

        if np.random.rand() < self.exploration_prob:
            action = sorted_options[randint(0, 2)]
            print("Random!")
        else:
            action = int(np.argmax(sum_state))

        print(f"Action: {final_state} = {sum_state} = {action}")

        return action

    def set_reward(
        self,
        states: List[Tuple],
        new_states: List[Tuple] | None,
        action: int,
        reward: float,
        stopped: bool,
    ):
        def __calc_reward(table: Dict, state: Tuple, table_idx: int) -> float:
            if stopped:
                score = (1 - self.learning_rate) * table[state][
                    action
                ] + self.learning_rate * reward
            else:
                assert new_states is not None

                new_state_scores = self.get_weights(new_states[table_idx], table_idx)

                score = (1 - self.learning_rate) * table[state][
                    action
                ] + self.learning_rate * (
                    reward + self.discount_factor * np.max(new_state_scores)
                )

            return score

        for i, (table, state) in enumerate(zip(self.Q_Table, states)):
            table[state][action] = __calc_reward(table, state, i)


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

            print(print_snake_vision(state.get_snake_vision(), state.snake_body[0]))

            max_steps = max(max_steps, step)
            max_length = max(max_length, len(state.snake_body))


            if tuple(current_state) not in Q_table:
                Q_table[tuple(current_state)] = np.zeros(4)

            print(current_state, Q_table[tuple(current_state)])


            action = tables.get_action(current_values)

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
                    reward = -0.1

            # if len(state.snake_body) == 10:
            #     need_stop = True
            #     reward = 10
            #     print("DING DING IDNG ")
            #     exit(1)

            if need_stop:
                score = (1 - learning_rate) * Q_table[tuple(current_state)][
                    action
                ] + learning_rate * reward
            else:
                new_state = compress_vision(
                        state.get_snake_vision(), state.snake_body[0]
                    )

                if tuple(new_state) not in Q_table:
                    Q_table[tuple(new_state)] = np.zeros(4)


            if need_stop:
                tables.set_reward(current_values, None, action, reward, need_stop)
                break

            next_values = compress_vision(state.get_snake_vision(), state.snake_body[0])

            tables.set_reward(current_values, next_values, action, reward, need_stop)

            step += 1
    print(max_length, max_steps)

