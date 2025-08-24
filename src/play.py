import asyncio
import dis
from time import sleep

from .gui import GameGUI, get_gradient_color
from .qtable_runner import QTableRunner, TrainState
from .train_state import Direction, print_snake_vision


async def start_display(display: GameGUI):
    while True:
        display.root.update()
        await asyncio.sleep(0.01)


async def play(model, visual: bool, board_size: int = 10):
    state = TrainState()
    table = QTableRunner()

    table.load_model(model)
    table.exploration_prob = 0
    table.explore_unseen = False
    # table.learning_rate = 0.2
    last_move = 0

    display = GameGUI(10)

    display_task = asyncio.create_task(start_display(display))

    while True:
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

        print_snake_vision(state.get_snake_vision(), state.snake_body[0], board_size)

        action = table.get_action(current_state)
        result = state.move_snake(Direction(action))
        last_move = action

        need_stop, reward = table.process_result(state, result)
        if need_stop:
            print("Died at length", len(state.snake_body))
            return

        next_values = table.compress_vision(
            state.get_snake_vision(), state.snake_body[0], last_move
        )

        table.set_reward(current_state, next_values, action, reward, need_stop)

        if visual:
            await asyncio.sleep(0.1)
