import argparse
import logging

from deepq.trainer import Trainer

logging.basicConfig(
    filename='./data/logs/training_log.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a DQN agent')
    parser.add_argument('-c', '--conf', nargs='?', default=None, help='training configuration file', dest='filename',
                        required=True)

    args = parser.parse_args()

    trainer = Trainer(args.filename)
    trainer.train()
