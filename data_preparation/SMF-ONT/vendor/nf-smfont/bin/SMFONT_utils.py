import tqdm
import re
from pathlib import Path
import argparse
import gzip
import numpy as np
import tabix
import pyfaidx

class FileType(argparse.FileType):
    """
    Overriding argparse.FileType to support gzipped files.
    """
    def __call__(self, string):
        if "r" in self._mode and any(string.endswith(ext) for ext in [".gz", ".bgz"]):
            return gzip.open(string, self._mode, self._bufsize, self._encoding, self._errors)
        return super().__call__(string)


def sequence_complement(sequence: str):
    sequence_complement_dict = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    sequence = "".join([sequence_complement_dict[b] for b in sequence])
    return sequence


def reverse_complement(sequence: str):
    return sequence_complement(sequence)[::-1]


def chunk_input_genome(fasta, chunk_size):
    ref = pyfaidx.Fasta(fasta)
    out = []
    for chr in ref.keys():
        chr_len=len(ref[chr])
        if chr_len <= chunk_size:
            out.append([chr, 0, len(ref[chr])])
        else:
            for i in range(0, chr_len, int(chunk_size)):
                if i + int(chunk_size) - 1 > chr_len:
                    end = chr_len
                else:
                    end = i + int(chunk_size) - 1
                out.append([chr, i, end])
    return out


class AccessibilityEntry:
    def __init__(
        self,
        chrom: str,
        start: int,
        end: int,
        fwd_strand: bool,
        read_name: str,
        llrs: np.ndarray,
    ):
        self.chrom = chrom
        self.read_name = read_name
        self.start = start
        self.end = end
        self.fwd_strand = fwd_strand
        self.llrs = llrs
    def __str__(self):
        code = "_.-°¯"
        curve = "".join([code[max(0, min(len(code) - 1, int(round(x / 2) + 2)))] for x in self.llrs])
        return "%s%s" % ("+" if self.fwd_strand else "-", curve)

    def __repr__(self):
        return str(self)


def parse_llrs_string(llrs_string):
    llrs = np.array([x for x in llrs_string.encode("ascii")], dtype=float) # maps range [33,125] to the range [-20, 20]
    llrs = llrs - 79
    llrs = (np.exp(np.abs(llrs) / 15.0) - 1) * np.sign(llrs)
    return llrs


def parse_line(line):
    line = line.split("\t")
    assert line[1] == "+" or line[1] == "-"
    if len(line) != 6:
        print(line)
        return None
    chrom = line[0]
    fwd_strand = line[1] == "+"
    start = int(line[2])
    end = int(line[3])
    read_name = line[4]
    llrs = parse_llrs_string(line[5])
    entry = AccessibilityEntry(
        chrom=chrom,
        start=start,
        end=end,
        read_name=read_name,
        fwd_strand=fwd_strand,
        llrs=llrs,
    )
    return entry


class Converter:
    def __init__(
        self,
        reference_fasta_file,
        llr_threshold=2,
        flank=2,
        motifs={"CG", "GC"},
        exclude_zero_calls=False,
        offset_gc_context=True,
    ):
        self.reference_fasta = reference_fasta_file
        self.llr_threshold = llr_threshold
        self.flank = flank
        self.motifs = motifs
        self.exclude_zero_calls = exclude_zero_calls
        self.canoncal_bases = set()
        self.offset_gc_context = offset_gc_context
        if "CG" in motifs:
            self.canoncal_bases.add("C")
        if "C" in motifs:
            self.canoncal_bases.add("C")
        if "GC" in motifs:
            self.canoncal_bases.add("C")
        if "A" in motifs:
            self.canoncal_bases.add("A")

    def __enter__(self):
        self.ref = pyfaidx.Fasta(self.reference_fasta)
        self.ref.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self.ref.__exit__(*args, **kwargs)

    def get_mod_character(self, llr):
        return ("m" if llr > 0 else "u") if abs(llr) > self.llr_threshold else "x"

    def convert_line(self, line: AccessibilityEntry):
        chrom = self.ref[line.chrom]
        chrom_len = len(chrom)

        if line.start <= self.flank: # we can't work with stuff right at the beginning of the contig
            line.llrs = line.llrs[self.flank + 1 - line.start:]
            line.start = self.flank + 1

        seq = str(chrom[line.start - self.flank - 1 : line.end + self.flank + 1]).upper()

        timp_line = {
            "chrom": line.chrom,
            "start": f"{line.start}",
            "end": f"{line.end}",
            "read_name": line.read_name,
            "cigmr": [],
            "llrs": [],
            "context": [],
        }
        space = 0
        for i in range(len(line.llrs)):
            if line.llrs[i] == 0 and self.exclude_zero_calls:
                space += 1
                continue

            seq_i = i + self.flank + 1  # because we extracted the sequence with a window
            if seq_i + self.flank + 1 >= chrom_len or seq_i >= len(seq):
                break

            base = seq[seq_i]
            if not line.fwd_strand:
                base = reverse_complement(base)

            if base not in self.canoncal_bases:
                space += 1
                continue

            threemer = seq[seq_i - 1 : seq_i + 2]
            if not line.fwd_strand:
                threemer = reverse_complement(threemer)

            if threemer[:2] == "GC" and "GC" in self.motifs: # Special treatment for GCs because we have somehow settled on centering the G in the context
                if self.offset_gc_context:
                    if line.fwd_strand:
                        context = seq[seq_i - self.flank - 1 : seq_i + self.flank]
                    else:
                        context = seq[seq_i - self.flank + 1 : seq_i + self.flank + 2]
                else:
                    context = seq[seq_i - self.flank : seq_i + self.flank + 1]
            elif threemer[1:] == "CG" and "CG" in self.motifs or threemer[1] == "A" and "A" in self.motifs or threemer[1] == "C" and "C" in self.motifs:
                context = seq[seq_i - self.flank : seq_i + self.flank + 1]
            else:
                context = ""

            if len(context) != self.flank * 2 + 1:
                space += 1
                continue

            if not line.fwd_strand:
                context = reverse_complement(context)

            call = self.get_mod_character(line.llrs[i])
            timp_line["cigmr"].append(f"{space}")
            timp_line["cigmr"].append(call)
            timp_line["context"].append(context)
            timp_line["llrs"].append(f"{line.llrs[i]:.2f}")
            space = 1

        timp_line["cigmr"] = "".join(timp_line["cigmr"])
        timp_line["context"] = ",".join(timp_line["context"])
        timp_line["llrs"] = ",".join(timp_line["llrs"])

        if len(timp_line["llrs"]) == 0: # There were no motifs
            return None

        return timp_line
    def format_line(self, line_dict):
        if line_dict is None:
            return None
        return "\t".join(
            [
                line_dict["chrom"],
                line_dict["start"],
                line_dict["end"],
                line_dict["read_name"],
                line_dict["cigmr"],
                line_dict["llrs"],
                line_dict["context"],
            ]
        )
    def convert_lines(self, in_stream, has_header=True):
        if has_header:
            in_stream.readline()

        for line in in_stream.readlines():
            line = line.strip()
            try:
                entry = parse_line(line)
                out_entry = self.convert_line(entry)
                if out_entry is None:
                    continue

                if len(out_entry["context"]) == 0: # Don't write file if the thresholding made it so that there isn't a single confident call
                    continue

                yield self.format_line(out_entry)
            except:
                pass

    def convert_file(self, in_file_name, out_file_name):
        if out_file_name.endswith(".gz") or out_file_name.endswith(".bgz"):
            ofn = gzip.open
        else:
            ofn = open

        with open(in_file_name, "rt") as in_f, ofn(out_file_name, "wt") as out_f:
            for line in self.convert_lines(in_f):
                print(line, file=out_f)

    def convert_files(self, in_file_names, out_file_name):
        with open(out_file_name, "w") as out_f:
            for in_file_name in in_file_names:
                with open(in_file_name, "rt") as in_f:
                    for line in self.convert_lines(in_f):
                        print(line, file=out_f)


class TimpFormatReader:
    def __init__(self, filename):
        if not filename.endswith("gz"):
            raise ValueError(f"{filename} must be bgzipped file")
        if not Path(filename).exists() or not Path(filename).is_file():
            raise ValueError(f"{filename} is not a valid file")
        index_filename = f"{filename}.tbi"
        if not Path(index_filename).exists():
            raise ValueError(f"{filename} must be indexed. Cannot find index file {index_filename}")
        self.filename = filename
        self.fp = tabix.open(self.filename)
    def parse_met_line(self, line, min_interested=0, max_interested=int(1e16)):
        coords_met = []
        coords_unmet = []
        pos = line["start"]
        for token in re.split("(x|u|m)", line["call_string"])[:-1]:
            move = 0
            if token == "m":
                coords_met.append(pos)
            elif token == "u":
                coords_unmet.append(pos)
            elif token != "x":
                move = int(token)
            pos += move
            if pos > max_interested:
                break
        if min_interested > 0:
            coords_met = [c for c in coords_met if c >= min_interested]
            coords_unmet = [c for c in coords_unmet if c >= min_interested]
        return np.array(coords_met, dtype=int), np.array(coords_unmet, dtype=int)
    def fetch(self, chrom: str, start: int, end: int, parse=False):
        try:
            for line in self.fp.query(chrom, start, end):
                if parse:
                    yield dict(
                        chrom=line[0], start=int(line[1]), end=int(line[2]), read_name=line[3], call_string=line[4]
                    )
                else:
                    yield line
        except:
            print(chrom, "not in file")
    def compute_met_unmet_counts(self, chrom: str, start: int, end: int):
        met = np.zeros(end - start)
        unmet = np.zeros(end - start)
        for line in self.fetch(chrom, start, end, parse=True):
            def increment(to_increment, coords):
                coords -= start
                coords = coords[(coords >= 0) & (coords < end - start)]
                to_increment[coords] += 1
            met_line, unmet_line = self.parse_met_line(line, max_interested=end)
            increment(met, met_line)
            increment(unmet, unmet_line)
        return met, unmet
    def compute_met_rate(self, coord_iterator, progress=False):
        met = None
        unmet = None
        for coord in tqdm.tqdm(list(coord_iterator), disable=not progress):
            met_part, unmet_part = self.compute_met_unmet_counts(coord["chrom"], coord["start"], coord["end"])
            if met is None:
                met = met_part
                unmet = unmet_part
            met += met_part
            unmet += unmet_part
        return met / (met + unmet)


def llr_to_p(llr, prior=0.5):
    """
    Convert log-likelihood ratios log(p(x|a)/p(x|~a)) to posterior
    probabilty p(a|x) given a prior p(a). For unbiased prediction,
    leave prior at 0.5
    """
    return 1 / (1 + np.exp(-llr) * (1 / prior - 1))


def p_to_llr(p, prior=0.5, eps=1e-16):
    """
    Converts the posterior probability p(a|x) into a log-likelihood ratio
    log(p(x|a)/p(x|~a)) given a prior p(a)
    """
    p = np.clip(p, eps, 1-eps) # Avoid numerical errors when p is 0 or 1
    return -np.log(prior * (1 - p) / (p * (1 - prior)))


def llr_to_uncertainty(llr, method="linear"):
    if method == "linear":
        p = llr_to_p(llr)
        return 0.5 - np.abs(0.5 - p)
