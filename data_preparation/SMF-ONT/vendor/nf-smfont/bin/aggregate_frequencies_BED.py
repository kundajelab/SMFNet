#! /usr/bin/env python3
import json
import sys
import argparse
import gzip
from importlib.metadata import version

class FileType(argparse.FileType):
    # Overriding argparse.FileType to support gzipped files
    def __call__(self, string):
        if "r" in self._mode and any(string.endswith(ext) for ext in [".gz", ".bgz"]):
            return gzip.open(string, self._mode, self._bufsize, self._encoding, self._errors)
        return super().__call__(string)


def parse_args():
    parser = argparse.ArgumentParser(description="Reads sorted bed file aggregates frequencies")
    parser.add_argument("-i", "--input", type=FileType("rt"),
        default=sys.stdin, help="input bed file (default stdin) - must be sorted")
    parser.add_argument( "-o", "--output", type=argparse.FileType("w"),
        default=sys.stdout, help="output file path (default stdout)")
    parser.add_argument("-s", "--skiplines", type=int, default=0, help="Header lines. One will be included in output")
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


def sort_key(parsed_line):
    return (line[0], int(line[1]))


def parse_line(line):
    line = line.strip().split("\t")
    return {
        "chrom": line[0],
        "start": int(line[1]),
        "end": int(line[2]),
        "called_sites": int(line[3]),
        "called_sites_methylated": int(line[4]),
        "context": line[6],
    }


def write(cur_line,outputfile):
    freq = f"{cur_line['called_sites_methylated'] / cur_line['called_sites']:.5f}"
    to_write = [
        cur_line["chrom"],
        str(cur_line["start"]),
        str(cur_line["end"]),
        str(cur_line["called_sites"]),
        str(cur_line["called_sites_methylated"]),
        freq,
        cur_line["context"],
    ]
    print("\t".join(to_write), flush=outputfile == sys.stdout, file=outputfile)


def reduce(inputfile, outputfile, header_lines):
    # making sure it is ordered
    for _ in range(header_lines):
        # Print header
        print(next(inputfile).strip(), flush=outputfile == sys.stdout, file=outputfile)
    
    try:
        cur_line = parse_line(next(inputfile))
        cur_key = (cur_line["chrom"], cur_line["start"])
        while True:
            line = parse_line(next(inputfile))
            key = (line["chrom"], line["start"])
            if key == cur_key:
                for k in "called_sites", "called_sites_methylated":
                    cur_line[k] += line[k]
            else:
                write(cur_line, outputfile)
                if key <= cur_key:
                    raise ValueError(
                        "Input file does not seem to be sorted. We were already at", cur_key, "but now read", key
                    )
                cur_line = line
                cur_key = key
    except StopIteration:
        # done
        write(cur_line, outputfile)
        pass


def main():
    args = parse_args()
    reduce(args.input, args.output, args.skiplines)
    get_version()

if __name__ == "__main__":
    main()
