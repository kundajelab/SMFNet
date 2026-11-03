#! /usr/bin/env python3
import json
import sys
import argparse
import gzip
import heapq
from importlib.metadata import version


class FileType(argparse.FileType):
    # Overriding argparse.FileType to support gzipped files.
    def __call__(self, string):
        if "r" in self._mode and any(string.endswith(ext) for ext in [".gz", ".bgz"]):
            return gzip.open(string, self._mode, self._bufsize, self._encoding, self._errors)
        return super().__call__(string)


def parse_args():
    parser = argparse.ArgumentParser(description="Reads sorted bed file and merges them in a sorted fashion")
    parser.add_argument("-i", "--input", type=FileType("rt"), nargs="*", 
        required=True, help="input bed files - must be sorted")
    parser.add_argument("-o", "--output", type=argparse.FileType("w"),
        default=sys.stdout, help="output file path (default stdout)")
    parser.add_argument("-s", "--skiplines", type=int,
        default=0, help="Header lines. One will be included in output")
    args = parser.parse_args()
    return args


def get_version():
    modulenames = set(sys.modules) & set(globals())
    with open('python_packages.txt', 'a') as f:
        for m in modulenames:
            try:
                f.write(f"    python-{m}: {globals()[m].__version__}\n")
            except AttributeError:
                try:
                    f.write(f"    python-{m}: {version(m)}\n")
                except ModuleNotFoundError:
                    f.write(f"    python-{m}: ????\n")


def sort_key(line):
    line = line.strip().split("\t", 3)
    # we dont need to parse more than the first 2 columns, since thats what we use for sorting
    return (line[0], int(line[1]))


def iterable_filepointer(inputfile):
    for line in inputfile:
        yield (sort_key(line), line.strip())


def merge(inputfiles, outputfile, header_lines):
    # making sure it is ordered
    inputfiles = list(inputfiles)
    try:
        for _ in range(header_lines):
            for i, f in enumerate(inputfiles):
                if i == 0:
                    # Print first header
                    print(next(f).strip(), flush=outputfile == sys.stdout, file=outputfile)
                else:
                    # Discard other headers
                    f.readline()
    except StopIteration:
        # File is only header
        raise ValueError("One of the files is empty or contains only a header")
    
    inputfiles = [iterable_filepointer(f) for f in inputfiles]
    for line in heapq.merge(*inputfiles, key=lambda line: line[0]):
        print(line[1], flush=outputfile == sys.stdout, file=outputfile)


def main():
    args = parse_args()
    merge(args.input, args.output, args.skiplines)
    get_version()


if __name__ == "__main__":
    main()
