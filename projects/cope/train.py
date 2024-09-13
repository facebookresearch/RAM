from argparse import ArgumentParser

from src.main import add_train_args, train

if __name__ == "__main__":
    parser = ArgumentParser()
    add_train_args(parser)
    cfg = parser.parse_args()
    train(cfg)
