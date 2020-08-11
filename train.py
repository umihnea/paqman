import argparse
import logging

from common.trainer import Trainer
from per.trainer import PERTrainer

logging.basicConfig(
    filename="./data/logs/training.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent")
    parser.add_argument(
        "-c",
        "--conf",
        nargs="?",
        default=None,
        help="training configuration file",
        dest="filename",
        required=True,
    )
    parser.add_argument(
        "--per", action="store_true", help="use prioritized experience replay"
    )
    parser.add_argument("--doubledqn", action="store_true", help="use double DQN")

    args = parser.parse_args()

    if args.doubledqn:
        raise NotImplementedError("Double DQN not yet implemented.")
    elif args.per:
        trainer = PERTrainer(args.filename)
        trainer.train()
    else:
        trainer = Trainer(args.filename)
        trainer.train()
