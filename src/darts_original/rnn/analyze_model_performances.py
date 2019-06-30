import numpy as np
import os
import re
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--prefix", default="eval-EXP-random-arc")


def parse_performance(path):
    f = open(path, "r")

    contents = f.read()
    line = re.search("test ppl.*", contents)

    if line is None:
        return None
    else:
        temp = line.group(0)
        temp = temp.split(" ")[-1]

        return float(temp)


def main(args):
    dirs = os.listdir(".")
    performances = []

    for d in dirs:
        if os.path.isdir(d) and d.startswith(args.prefix):
            performance = parse_performance(os.path.join(d, "log.txt"))

            if performance is not None:
                performances.append(performance)

    print("{} model performances for {} models".format(args.prefix, len(performances)))
    print("Min: {} | Max: {}".format(np.min(performances), np.max(performances)))
    print("Mean: {}".format(np.mean(performances)))
    print("std: {}".format(np.std(performances)))


if __name__ == "__main__":
    main(parser.parse_args())