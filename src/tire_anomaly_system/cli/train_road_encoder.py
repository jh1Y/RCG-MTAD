"""Standalone road-encoder training using the prepared four-state data."""

from .train_model import parse_args, train


def main():
    train(parse_args(stage="road"), stage="road")


if __name__ == "__main__":
    main()
