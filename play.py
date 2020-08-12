import argparse
import logging

from common.evaluator import Evaluator

logging.basicConfig(
    filename="./data/logs/evaluation.log",
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

    args = parser.parse_args()

    evaluator = Evaluator(args.config, args.checkpoint)
    evaluator.evaluate()
