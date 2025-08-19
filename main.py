import argparse
from src.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Learn2Slither CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--sessions", type=int, required=True, help="Number of training sessions"
    )
    train_parser.add_argument(
        "--visual", action="store_true", help="Enable visual mode during training"
    )

    play_parser = subparsers.add_parser("play", help="Play the game")
    play_parser.add_argument(
        "--visual", action="store_true", help="Enable visual mode during play"
    )

    play_parser.add_argument(
        "model", type=argparse.FileType("r"), help="Model to be loaded"
    )

    args = parser.parse_args()
    print(args)

    match args.command:
        case "train":
            train_model(args.sessions, args.visual)


if __name__ == "__main__":
    main()
