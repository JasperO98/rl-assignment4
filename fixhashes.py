import setup
from glob import glob
from os.path import join, sep, exists
from os import makedirs, rmdir
import json
from hex.players.selfplay import ArgsCoach
from shutil import move


def remove_empty_folders(root):
    for folder in glob(join(root, '*/')):
        remove_empty_folders(folder)

    if len(glob(join(root, '*'))) == 0:
        rmdir(root)


if __name__ == '__main__':
    for path in glob(join('models', '*', '*', '*')):
        args = ArgsCoach()

        with open(join(path, 'parameters.json'), 'r') as fp:
            for key, value in json.load(fp).items():
                args.__setattr__(key, value)

        separated = path.split(sep)
        moveto = join(separated[0], separated[1], str(hash(args)))

        if exists(join(moveto, separated[3])):
            continue

        makedirs(name=moveto, exist_ok=True)
        move(path, moveto)

    remove_empty_folders('models')
