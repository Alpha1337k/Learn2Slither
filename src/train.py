from email.policy import default
from itertools import count
import json
from operator import index
from random import randint, random
from re import A
from typing import Dict, List, Tuple
from typing_extensions import Self
from pydantic import ConfigDict, validate_call
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .qtable_runner import QTableRunner
from .train_state import TrainState, print_snake_vision, BoardPiece, Direction


model_config = ConfigDict(arbitrary_types_allowed=True)


@validate_call(config=model_config)
def train_model(epochs: int, visual: bool):
    table = QTableRunner(3)

    max_steps = 0
    max_length = 0

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

            current_state = table.compress_vision(
                state.get_snake_vision(), state.snake_body[0]
            )

            print_snake_vision(state.get_snake_vision(), state.snake_body[0])
            print(current_state)

            max_steps = max(max_steps, step)
            max_length = max(max_length, len(state.snake_body))

            action = table.get_action(current_state)

            reward = 0
            result = state.move_snake(Direction(action))

            need_stop, reward = table.process_result(state, result)

            if need_stop:
                table.set_reward(current_state, None, action, reward, need_stop)
                break

            next_values = table.compress_vision(
                state.get_snake_vision(), state.snake_body[0]
            )

            table.set_reward(current_state, next_values, action, reward, need_stop)

            step += 1
    print(max_length, max_steps)

    table.save_model(f"models/model_{epochs}.json")
