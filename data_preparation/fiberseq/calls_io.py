"""Read and write calls files (``<chrom>.tsv.zst``, format in README.md).

Compression goes through the ``zstd`` command-line tool in a subprocess. Long mode
(``--long=27``, a 128 MB window) matters: overlapping molecules repeat the no-call pattern
the reference fixes at C/G bases, and long mode compresses GM12878 chr21 15x against 6x
without it. Files written with it must be read with the same flag.
"""

import os
import subprocess
from contextlib import contextmanager

SUFFIX = ".tsv.zst"
ZSTD_LEVEL = 12
ZSTD_LONG = "--long=27"
COLUMNS = ("chrom", "start", "end", "read_id", "strand", "haplotype", "calls")


def check_path(path):
    if not str(path).endswith(SUFFIX):
        raise ValueError(f"calls files must end in {SUFFIX}, got {path}")


@contextmanager
def write_calls(path, threads=1):
    """Yield a text handle; the file only appears at `path` once zstd has finished.

    Writing goes to ``<path>.tmp`` first, so a preempted or failed run never leaves a
    truncated file under the final name.
    """
    check_path(path)
    tmp = f"{path}.tmp"
    proc = subprocess.Popen(
        ["zstd", "-q", "-f", f"-{ZSTD_LEVEL}", ZSTD_LONG, f"-T{threads}", "-o", tmp],
        stdin=subprocess.PIPE,
        text=True,
        bufsize=1 << 20,
    )
    try:
        yield proc.stdin
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"zstd failed writing {path} ({proc.returncode})")
        os.replace(tmp, path)
    except BaseException:
        if not proc.stdin.closed:
            proc.stdin.close()
        proc.wait()
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def iter_calls(path):
    """Yield the tab-split fields of every non-comment row, streaming through zstd."""
    check_path(path)
    proc = subprocess.Popen(
        ["zstd", "-dc", "-q", ZSTD_LONG, str(path)],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1 << 20,
    )
    try:
        for line in proc.stdout:
            if not line.startswith("#"):
                yield line.rstrip("\n").split("\t")
    finally:
        proc.stdout.close()
        returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"zstd failed reading {path} ({returncode})")
