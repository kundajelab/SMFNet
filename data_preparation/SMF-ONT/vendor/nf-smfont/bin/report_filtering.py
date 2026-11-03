#! /usr/bin/env python
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate read methylation rate json files"
    )
    parser.add_argument("-i", "--input", type=str, nargs='+',
        required=True, help="input methylation json files")
    parser.add_argument("-o", "--output", type=str,
        required=True, help="output json file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    total_stats = {}
    for file in args.input:
        with open(file, "rt") as f:
            part_dict = json.load(f)
            for k,v in part_dict.items():
                total_stats[k] = total_stats.get(k, 0) + v
    
    with open(args.output, "wt") as f:
        json.dump(total_stats, f)


if __name__ == "__main__":
    main()
