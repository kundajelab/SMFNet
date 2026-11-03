#! /usr/bin/env python3
import os
import math
import bisect
import sys
import argparse
import gzip
import re
import numpy as np
from collections import namedtuple

from pyfaidx import Fasta
from SMFONT_utils import reverse_complement
from importlib.metadata import version

def parseArgs():
    parser = argparse.ArgumentParser(description="Create bed file containing methylation frequency per site")
    parser.add_argument("-v", "--verbose", action="store_true", required=False,
        default=False, help="verbose output")
    parser.add_argument("-i", "--input", type=str, required=False,
        help="input methylation bed file (default stdin)")
    parser.add_argument("-o", "--out", type=argparse.FileType("w"), required=False, 
        default=sys.stdout, help="output file path (default stdout)")
    parser.add_argument("-c", "--collapse_strands", action="store_true", required=False,
        help="Collapses symmetrical motifs such as CpG and GpC")
    parser.add_argument("-r", "--ref_fasta", required=False,
        help="Reference fasta file")
    args = parser.parse_args()
    if args.collapse_strands and "ref" in vars(args):
        raise ValueError("Requires fasta file in order to collapse strand")
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
    def __init__(self, string, ref, collapse_strands=False):
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
        self.calldict = self.parseMeth(ref, collapse_strands=collapse_strands)
        self.keys = sorted(self.calldict.keys())
        self.callarray = self.getArray(self.calldict)
            
    def determine_strand(self, pos, seq, ref):
        if ref is None:
            return "+"
        """Determine strand by comparing first motif of read with reference"""
        ref_base = ref[self.rname][pos -1 : pos + 2].seq.upper()
        tricontext = get_tricontext(seq)
        if ref_base == tricontext:
            return "+"
        elif ref_base == reverse_complement(tricontext):
            return "-"
        else:
            raise ValueError(f"Motif string in input file {seq} does not correspond reference sequence {ref_base} at {pos}")
        
    def make_key(self, pos):
        return int(pos)
    
    def parseMeth(self, ref, collapse_strands=False):
        calldict = dict()
        pos = int(self.start)
        calls = re.findall("(\d+)([umx])?", self.methstring)
        for i, (dist, call) in enumerate(calls):
            pos += int(dist)
            if call == "x":
                is_meth = -1
            else:
                is_meth = int(call == "m")

            if i == 0:
                # determine strand for strand collapsing
                strand = self.determine_strand(pos, self.seqs[i], ref)
                
            offset = 0
            if collapse_strands and strand == "-":
                """Optional strand collapsing for GC and CG"""
                if self.seqs[i][2:4] == "GC":
                    offset = 1
                elif self.seqs[i][2:4] == "CG":
                    offset = -1
                    
                if offset != 0:
                    self.seqs[i] = reverse_complement(self.seqs[i])
                    
            calldict[self.make_key(pos+offset)] = methylCall(pos+offset, is_meth, float(self.ratios[i]), self.seqs[i])
        return calldict
    
    def getArray(self, calldict):
        callarray = np.array([(x, calldict[x].call) for x in sorted(calldict.keys())])
        return callarray


def get_tricontext(seq):
    motifCenterid = int((len(seq) - 1) / 2)
    if seq[motifCenterid] == "G":
        return seq[motifCenterid: motifCenterid + 3]
    elif seq[motifCenterid] in {"C", "A"}:
        return seq[motifCenterid - 1: motifCenterid + 2]
    else:
        raise ValueError("Tricontext does not match any known motif!")


class SiteStats:
    def __init__(self, methcall, chrom):
        self.rname = chrom
        self.pos = methcall.pos
        self.num_reads = 0
        self.num_methylated = 0
        self.seq = methcall.seq
    
    def update(self, methcall):
        if methcall.call == -1:
            return
        self.num_reads += 1
        self.num_methylated += methcall.call
    
    def printFreq(self, out):
        if self.num_reads == 0:
            return
        
        self.tricontext = get_tricontext(self.seq)
        print(
            "\t".join(
                [
                    str(x)
                    for x in [
                        self.rname,
                        self.pos + 1,
                        self.pos + 1,
                        self.num_reads,
                        self.num_methylated,
                        round(self.num_methylated / self.num_reads, 5),
                        self.tricontext,
                    ]
                ]
            ),
            file=out,
        )


def getFreq(args, in_fh):
    if args.ref_fasta is not None:
        ref = Fasta(args.ref_fasta, "r")
    else:
        ref = None
    
    if args.verbose:
        print("getting methylation frequency", file=sys.stderr)
    sites = dict()
    n = 0
    print(
        "\t".join(
            ["chr", "start", "end", "called_sites", "called_sites_methylated", "methylated_frequency", "group_sequence"]
        ),
        file=args.out,
    )
    for line in in_fh:
        try:
            line = line.decode("ascii")
        except AttributeError:
            pass
        n += 1
        read = MethRead(line, ref, args.collapse_strands)
        sitekeys = sorted(sites.keys())
        try:
            if read.rname != sites[sitekeys[0]].rname:  # print everything in sites if chromosome is different
                printind = len(sitekeys)
            else:
                printind = bisect.bisect_left(sitekeys, read.keys[0])  # get index of new position
        except IndexError:
            printind = 0
        if printind != 0:
            for i in range(printind):
                key = sitekeys[i]
                sites[key].printFreq(args.out)
                sites.pop(key)
        for key in read.keys:
            if key not in sites.keys():
                sites[key] = SiteStats(read.calldict[key], read.rname)
            sites[key].update(read.calldict[key])
        if args.verbose:
            if n % 10000 == 0:
                print("parsed {} lines".format(n), file=sys.stderr)
    for key in sorted(sites.keys()):
        sites[key].printFreq(args.out)


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
    getFreq(args, in_fh)
    in_fh.close()
    get_version()


if __name__ == "__main__":
    main()
