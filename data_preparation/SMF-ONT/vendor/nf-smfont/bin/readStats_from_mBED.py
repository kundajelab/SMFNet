#! /usr/bin/env python3
import os
import sys
import argparse
import gzip
import math
import bisect
import re
import numpy as np
from collections import namedtuple
from importlib.metadata import version


def parseArgs():
    parser = argparse.ArgumentParser(description="Get reduce bed file with read names and read size")
    parser.add_argument("-v", "--verbose", action="store_true", required=False,
        default=False, help="verbose output")
    parser.add_argument("-i", "--input", type=str, required=False,
        help="input methylation bed file (default stdin)")
    parser.add_argument("-e", "--exclude", type=str, required=False,
        default="", help="motif to exclude (default : none)")
    parser.add_argument("-o", "--out", type=argparse.FileType("w"), required=False,
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


# https://stackoverflow.com/questions/107705/disable-output-buffering
class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    
    def __getattr__(self, attr):
        return getattr(self.stream, attr)


methylCall = namedtuple("methylCall", ["pos", "call", "ratio", "seq"])


class MethRead:
    def __init__(self, string):
        self.string = string
        self.fields = string.strip().split("\t")
        self.rname = self.fields[0]
        self.start = int(self.fields[1])
        self.end = int(self.fields[2])
        self.rlen = self.end - self.start
        self.qname = self.fields[3]
        self.methstring = self.fields[4]
        self.ratios = self.fields[5].strip().split(",")
        self.seqs = self.fields[6].strip().split(",")
        self.calldict = self.parseMeth()
        self.keys = sorted(self.calldict.keys())
        self.callarray = self.getArray(self.calldict)
        self.motif = self.parseMotif()
    
    def make_key(self, pos):
        return int(pos)
    
    def parseMeth(self):
        calldict = dict()
        pos = int(self.start)
        calls = re.findall("(\d+)([umx])?", self.methstring)
        for i, (dist, call) in enumerate(calls):
            pos += int(dist)
            if call == "x":
                is_meth = -1
            else:
                is_meth = int(call == "m")
            calldict[self.make_key(pos)] = methylCall(pos, is_meth, float(self.ratios[i]), self.seqs[i])
        return calldict
    
    def getArray(self, calldict):
        callarray = np.array([(x, calldict[x].call) for x in sorted(calldict.keys())])
        return callarray
    
    def parseMotif(self):
        firstSeq = self.seqs[0]
        if firstSeq[2] == "C":
            return "CG"
        elif firstSeq[2] == "G":
            return "GC"
        elif firstSeq[2] == "A":
            return "A"
        else:
            return "NA"
    
    def printInfo(self, out, calls, under):
        if under:
            meth = "NA"
            numCall = "NA"
        else:
            meth = np.mean(calls)
            numCall = len(calls)
        print(
            "\t".join(
                [
                    str(x)
                    for x in [
                        self.rname,
                        self.start + 1,
                        self.end + 1,
                        self.qname,
                        self.rlen,
                        meth,
                        numCall,
                        self.motif,
                    ]
                ]
            ),
            file=out,
        )


def getReadStats(args, in_fh):
    if args.verbose:
        print("getting read stats...", file=sys.stderr)
    n = 0
    print(
        "\t".join(["chr", "start", "end", "readID", "read_length", "read_methylation_rate", "called_Cs", "context"]),
        file=args.out,
    )
    for line in in_fh:
        try:
            line = line.decode("ascii")
        except AttributeError:
            pass
        n += 1
        read = MethRead(line)
        seqs = [x.seq for x in read.calldict.values()]
        filteridx = [idx for idx, s in enumerate(seqs) if args.exclude not in s]  # filter out calls with unwanted seq
        if len(filteridx) != 0:
            calls_all = np.array([x.call for x in read.calldict.values()])[filteridx]
            ratios = np.array([x.ratio for x in read.calldict.values()])[filteridx]
        else:
            calls_all = np.array([x.call for x in read.calldict.values()])
            ratios = np.array([x.ratio for x in read.calldict.values()])
        calls = [x for x in calls_all if x != -1]  # filter out unconfident calls
        if len(calls) > 0:  # drop empty call strings
            read.printInfo(args.out, calls, under=False)
        else:
            read.printInfo(args.out, calls, under=True)
        if args.verbose:
            if n % 10000 == 0:
                print("parsed {} lines".format(n), file=sys.stderr)


def readInput(file):
    ext = file.strip().split(".")[-1]
    if ext in ["gz", "bgz"]:
        return gzip.open(file, "rt")
    else:
        return open(file, "r")


def main():
    args = parseArgs()
    if args.input:
        in_fh = readInput(args.input)
    else:
        in_fh = sys.stdin
    getReadStats(args, in_fh)
    in_fh.close()
    get_version()


if __name__ == "__main__":
    main()
