import asyncio
from email.policy import default
from itertools import count
import json
from operator import index
from random import randint, random
from re import A
from time import sleep
from typing import Dict, List, Tuple
from typing_extensions import Self
from pydantic import ConfigDict, validate_call
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .gui import GameGUI, get_gradient_color

from .qtable_runner import QTableRunner
from .train_state import TrainState, print_snake_vision, BoardPiece, Direction


model_config = ConfigDict(arbitrary_types_allowed=True)


@validate_call(config=model_config)
async def run_epoch(
    visual: bool, display: GameGUI, table: QTableRunner, board_size: int
):
    state = TrainState(board_size)

    assert len(state.green_apples) == 2
    assert len(state.red_apples) == 1

    max_steps = 0
    max_length = 0
    last_move = 0
    step = 0

    while step < 1000:
        display.clear_all()
        for apple in state.green_apples:
            display.fill_cell(apple[0], apple[1], "green")
        for apple in state.red_apples:
            display.fill_cell(apple[0], apple[1], "red")
        for i, segment in enumerate(state.snake_body):
            display.fill_cell(
                segment[0], segment[1], get_gradient_color(i, len(state.snake_body))
            )

        current_state = table.compress_vision(
            state.get_snake_vision(), state.snake_body[0], last_move
        )

        step += 1
        max_steps = max(max_steps, step)
        max_length = max(max_length, len(state.snake_body))

        print_snake_vision(state.get_snake_vision(), state.snake_body[0], board_size)

        action = table.get_action(current_state)
        result = state.move_snake(Direction(action))
        last_move = action

        need_stop, reward = table.process_result(state, result)
        if need_stop:
            table.set_reward(current_state, None, action, reward, need_stop)
            print("Died at length", len(state.snake_body))

            break

        else:
            next_values = table.compress_vision(
                state.get_snake_vision(), state.snake_body[0], last_move
            )

            table.set_reward(current_state, next_values, action, reward, need_stop)

        if visual:
            await asyncio.sleep(0.05)

    return max_length, max_steps


async def start_display(display: GameGUI):
    while True:
        display.root.update()
        await asyncio.sleep(0.01)


@validate_call(config=model_config)
async def train_model(epochs: int, visual: bool, board_size: int):
    table = QTableRunner(3)
    display = GameGUI(board_size)

    max_steps = 0
    max_length = 0

    np.set_printoptions(linewidth=np.inf)

    display_task = asyncio.create_task(start_display(display))
    await asyncio.sleep(0.1)

    for epoch in range(epochs):
        m_length, m_steps = await run_epoch(visual, display, table, board_size)
        max_length = max(max_length, m_length)
        max_steps = max(max_steps, m_steps)

    table.save_model(f"models/model_{epochs}_{board_size}.json")

    print(max_length, max_steps)
