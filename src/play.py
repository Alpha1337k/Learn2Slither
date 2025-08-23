import dis
from time import sleep

from .gui import GameGUI
from .qtable_runner import QTableRunner, TrainState
from .train_state import Direction, print_snake_vision


def get_gradient_color(index: int, max: int) -> str:
    ratio = index / max
    red = int(255 * (1 - ratio))
    green = int(255 * ratio)
    blue = int(255 * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


def play(model, visual: bool):
    state = TrainState()
    table = QTableRunner()

    table.load_model(model)
    table.exploration_prob = 0
    table.learning_rate = 0.1

    display = GameGUI(10)

    def run_step():
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
            state.get_snake_vision(), state.snake_body[0]
        )

        print_snake_vision(state.get_snake_vision(), state.snake_body[0])

        action = table.get_action(current_state)
        result = state.move_snake(Direction(action))

        need_stop, reward = table.process_result(state, result)
        if need_stop:
            print("Died at length", len(state.snake_body))
            return

        next_values = table.compress_vision(
            state.get_snake_vision(), state.snake_body[0]
        )

        table.set_reward(current_state, next_values, action, reward, need_stop)

        display.root.after(100, run_step)

    display.root.after(100, run_step)
    display.run()
