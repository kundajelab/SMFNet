#! /usr/bin/env python3
import json
import sys
import argparse
import gzip
from importlib.metadata import version


class FileType(argparse.FileType):
    #Overriding argparse.FileType to support gzipped files.
    def __call__(self, string):
        if "r" in self._mode and any(string.endswith(ext) for ext in [".gz", ".bgz"]):
            return gzip.open(string, self._mode, self._bufsize, self._encoding, self._errors)
        return super().__call__(string)


"""
Filters timp format files based on methylation rate and outputs readnames that passed the filter
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reads timp format files and outputs readnames of reads that pass filter criteria"
    )
    parser.add_argument("-l", "--lowerThreshold", type=float, required=False,
        default=0.05, help="Threshold for unmethylated reads filter (default : 0.05)")
    parser.add_argument("-u", "--upperThreshold", type=float, required=False,
        default=0.95, help="Threshold for fully methylated reads filter (default : 0.95)")
    parser.add_argument("-i", "--input", type=FileType("rt"),
        default=sys.stdin, help="input methylation bed file (default stdin)")
    parser.add_argument("-o", "--output", type=argparse.FileType("w"),
        default=sys.stdout, help="output file path (default stdout)")
    parser.add_argument("-r", "--report", type=argparse.FileType("w"),
        default=sys.stderr, help="Report file (default stderr)"
    )
    args = parser.parse_args()
    assert args.lowerThreshold is not None
    assert args.upperThreshold is not None
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


def count_met_unmet(in_fp):
    read_counts = {}
    for line in in_fp.readlines():
        line = line.split("\t")
        read = line[3]
        calls = line[4]
        # Count methylated and unmethylated
        counts = {t: calls.count(t) for t in "um"}
        # Access previous counts for read
        previous_counts = read_counts.get(read, {t: 0 for t in counts})
        # Add and store
        read_counts[read] = {t: counts[t] + previous_counts[t] for t in counts}
    return read_counts


def print_passing_reads(read_counts, min_rate, max_rate, out):
    stats = {"no_calls": 0, "too_low": 0, "too_high": 0, "pass": 0}
    
    for read, counts in read_counts.items():
        total_counts = counts["u"] + counts["m"]
        
        if total_counts == 0:
            stats["no_calls"] += 1
            continue
        
        rate = counts["m"] / total_counts
        
        if min_rate <= rate < max_rate:
            print(read, file=out, flush=True)
            stats["pass"] += 1
        elif rate < min_rate:
            stats["too_low"] += 1
        elif max_rate <= rate:
            stats["too_high"] += 1
    return stats


def main():
    args = parse_args()
    # Collect a dictionary of reads with methylated and unmethylated counts
    read_counts = count_met_unmet(args.input)
    # Filter reads and print those that passed the filter
    stats = print_passing_reads(read_counts, args.lowerThreshold, args.upperThreshold, args.output)
    # Print filter statistics
    json.dump(stats, args.report)
    get_version()


if __name__ == "__main__":
    main()
