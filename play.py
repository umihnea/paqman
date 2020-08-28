import argparse
import logging
import sys

from common.evaluator import Evaluator

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a DQN agent")
    parser.add_argument(
        "-c",
        "--conf",
        nargs="?",
        default=None,
        help="training configuration file",
        dest="config",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--file",
        nargs="?",
        default=None,
        help="checkpoint file",
        dest="checkpoint",
        required=True,
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="whether to just run a training with a random agent",
        dest="random",
    )

    args = parser.parse_args()

    random = bool(args.random)
    evaluator = Evaluator(args.config, args.checkpoint, random=random)
    evaluator.evaluate()
