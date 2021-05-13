from run import main
from onmt.utils.parse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(model="translate", description='translate.py')
    main(parser)
