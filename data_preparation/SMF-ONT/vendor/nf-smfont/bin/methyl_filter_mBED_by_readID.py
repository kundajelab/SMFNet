#! /usr/bin/env python3
import sys
import argparse
import gzip
from importlib.metadata import version


class FileType(argparse.FileType):
    # Overriding argparse.FileType to support gzipped files.
    def __call__(self, string):
        if "r" in self._mode and any(string.endswith(ext) for ext in [".gz", ".bgz"]):
            return gzip.open(string, self._mode, self._bufsize, self._encoding, self._errors)
        return super().__call__(string)


"""
Filters mBEDs file based on read names
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Filters timp format file based on readnames")
    parser.add_argument( "-i", "--input", type=FileType("rt"),
        default=sys.stdin, help="input methylation bed file (default stdin)")
    parser.add_argument( "-r", "--reads", type=FileType("rt"),
        required=True, help="file containing readnames to keep")
    parser.add_argument("-d", "--drop", action="store_true", required=False,
        default=False, help="instead of keeping reads from the file, drop them")
    parser.add_argument( "-o", "--output", type=argparse.FileType("w"),
        default=sys.stdout, help="output file path (default stdout)")
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


def main():
    args = parse_args()
    
    # Collect read names
    reads = {r.strip() for r in args.reads.readlines()}
    
    # Filter file
    for line in args.input.readlines():
        read = line.split("\t")[3]
        
        if (read in reads) != args.drop:
            print(line, file=args.output, flush=True, end="")

    get_version()


if __name__ == "__main__":
    main()
