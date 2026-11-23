"""Make a small demo data set from SMFNet training H5 files and their reference genome.

Samples ``--train-regions`` training-chromosome and ``--val-regions`` validation-chromosome
regions from each input H5 (none from the test chromosomes) and writes, to ``--out-dir``:

- one H5 per input, with the same name and layout (``smf_mat``, ``smf_pos``, ``read_id``)
  and the sampled regions' reads unchanged;
- ``demo.fa`` (and its ``.fai``), a small reference holding only the sampled regions'
  sequence, so the demo does not need the whole genome;
- ``regions.tsv``, mapping each demo region to its input file and original region.

In ``demo.fa`` each chromosome is one contig, keeping its name, made of its sampled regions
end to end. Regions are renamed to their place there (``chrom:start-end``, 1-based start,
exclusive end, as in the inputs), so the default chromosome splits still apply. Every demo
region is read back through ``smf_net.data.load_dna`` and checked against the original.
"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
from pyfaidx import Fasta

from smf_net.data import (
    DEFAULT_TRAIN_CHRM,
    DEFAULT_VAL_CHRM,
    load_dna,
    parse_region_string,
)

LINE_WIDTH = 60


def sample_regions(regions, chromosomes, n, rng):
    eligible = sorted(r for r in regions if parse_region_string(r)[0] in chromosomes)
    if len(eligible) < n:
        raise ValueError(f"Asked for {n} regions but only {len(eligible)} are eligible")
    return list(rng.choice(eligible, n, replace=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--h5", nargs="+", required=True, help="Input H5 files.")
    ap.add_argument("--fasta", required=True, help="Reference genome of the inputs.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-regions", type=int, default=750, help="Per input file.")
    ap.add_argument("--val-regions", type=int, default=150, help="Per input file.")
    ap.add_argument("--region-length", type=int, default=2114)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    genome = Fasta(args.fasta)

    # (input path, original region), in the order they go into the contigs.
    selected = []
    for path in args.h5:
        with h5py.File(path, "r") as h5:
            regions = list(h5["smf_mat"].keys())
        for chromosomes, n in [
            (DEFAULT_TRAIN_CHRM, args.train_regions),
            (DEFAULT_VAL_CHRM, args.val_regions),
        ]:
            selected += [(path, r) for r in sample_regions(regions, set(chromosomes), n, rng)]

    def contig_order(item):
        chrom, start, _ = parse_region_string(item[1])
        return chrom, args.h5.index(item[0]), start

    selected.sort(key=contig_order)

    contigs = {}
    new_names = {}
    for path, region in selected:
        chrom, start, end = parse_region_string(region)
        if end - start != args.region_length:
            raise ValueError(f"{region} does not span {args.region_length} bases")
        parts = contigs.setdefault(chrom, [])
        new_start = 1 + sum(len(p) for p in parts)
        new_names[path, region] = f"{chrom}:{new_start}-{new_start + args.region_length}"
        parts.append(genome[chrom][start - 1 : end - 1].seq)

    fasta_path = out_dir / "demo.fa"
    with open(fasta_path, "w") as fh:
        for chrom, parts in contigs.items():
            seq = "".join(parts)
            fh.write(f">{chrom}\n")
            fh.writelines(seq[i : i + LINE_WIDTH] + "\n" for i in range(0, len(seq), LINE_WIDTH))
    Fasta(str(fasta_path), rebuild=True)  # Writes the .fai.

    for path in args.h5:
        with h5py.File(path, "r") as src, h5py.File(out_dir / Path(path).name, "w") as dst:
            for (src_path, region), name in new_names.items():
                if src_path != path:
                    continue
                for group in ("smf_mat", "smf_pos", "read_id"):
                    data = src[group][region]
                    dst.create_dataset(
                        f"{group}/{name}",
                        data=data[()],
                        compression=data.compression,
                        compression_opts=data.compression_opts,
                    )

    with open(out_dir / "regions.tsv", "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["demo_region", "file", "original_region", "split"])
        for (path, region), name in new_names.items():
            split = "train" if parse_region_string(region)[0] in DEFAULT_TRAIN_CHRM else "val"
            writer.writerow([name, Path(path).name, region, split])

    original = load_dna(args.fasta, [r for _, r in selected], args.region_length)
    demo = load_dna(str(fasta_path), [new_names[k] for k in selected], args.region_length)
    if not torch.equal(original, demo):
        raise RuntimeError("Demo sequences differ from the original regions")
    print(f"Wrote {len(selected)} regions on {len(contigs)} contigs to {out_dir}")


if __name__ == "__main__":
    main()
