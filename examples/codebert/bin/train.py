from run import main
from onmt.utils.parse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(model="train", description='train.py')
    main(parser)
