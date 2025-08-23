import json
from random import randint
from typing import IO, List, Tuple

import numpy as np
from .train_state import BoardPiece, TrainState, Direction


class QTableRunner:
    def __init__(self, len=3):
        self.Q_Table = {}
        self.exploration_prob = 0.05
        self.learning_rate = 0.7
        self.discount_factor = 0.8

    def save_model(self, filename: str):
        with open(filename, "w") as f:
            json.dump(
                {str(k): v.tolist() for k, v in self.Q_Table.items()}, f, indent=4
            )

    def load_model(self, file: IO):
        data = json.load(file)
        self.Q_Table = {eval(k): np.array(v) for k, v in data.items()}

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

    def get_weights(self, state: Tuple[int, int]) -> np.ndarray:
        rotated_state = self.__rotate_state(state)
        if rotated_state not in self.Q_Table:
            self.Q_Table[rotated_state] = np.zeros(4)

        return self.Q_Table[rotated_state]

    def get_action(self, state: Tuple) -> int:
        weights = self.get_weights(state)
        sorted_options = np.argsort(weights)

        if 0.0 in weights:
            action = np.where(weights == 0.0)[0][0]
        elif np.random.rand() < self.exploration_prob:
            action = sorted_options[randint(1, 2)]
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

    def compress_vision(self, vision: List[List[BoardPiece]], pos: Tuple[int, int]):
        def get_distance(
            vision: List[List[BoardPiece]],
            direction: int,
            vertical: bool,
            item: BoardPiece,
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

        for direction, is_vertical in directions:
            dir_score = min(
                get_distance(vision, direction, is_vertical, BoardPiece.WALL)
                or compression,
                get_distance(vision, direction, is_vertical, BoardPiece.SNAKE)
                or compression,
                get_distance(vision, direction, is_vertical, BoardPiece.RED)
                or compression,
            )
            if (
                dir_score > 1
                and get_distance(vision, direction, is_vertical, BoardPiece.GREEN) > 0
            ):
                dir_score = 0
            data.append(dir_score)

        return tuple(data)

    def process_result(
        self, state: TrainState, result: BoardPiece
    ) -> Tuple[bool, float]:
        need_stop = False
        reward = 0

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
        return need_stop, reward
