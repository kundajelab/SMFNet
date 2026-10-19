# Notes for coding agents

## Environment and commands

The environment is managed with pixi (see README). Run everything through it:
`pixi run <command>` (GPU environment) or `pixi run -e cpu <command>`.

| Task | Does |
| --- | --- |
| `pixi run fmt` | Sort imports and format with ruff. Only cosmetic changes. |
| `pixi run lint` | Report ruff findings without changing anything. |
| `pixi run check` | Fail if anything is unformatted or has lint findings. |
| `pixi run test` | Run the tests. |

After editing, run `pixi run fmt`, then `pixi run check` and `pixi run test`.

Fix lint findings by hand, or with `ruff check --fix` after reading the diff.
Do not apply `--unsafe-fixes` without checking each change: some are wrong
for this code (e.g. iterating a pyfaidx `Fasta` yields records, not names).
Use `# noqa: <rule>` only for a deliberate exception on one line.

## Layout

- `src/smf_net/`: the package. `model.py` (SMFNet), `cnn.py`, `data.py`,
  `diffusion_schedule.py`, and `evaluation/` (sampling, metrics,
  marginalization, ISM, PRISM).
- `scripts/`: Hydra command-line apps (`train.py`, `sample.py`, `metrics.py`,
  `marginalize.py`, `ism.py`), run as `pixi run python scripts/<name>.py
  key=value ...`. Shared helpers are in `src/smf_net/cli.py`.
- `data_preparation/`: building training data from raw inputs.
- `tests/`: pytest tests.

## Conventions

- Published models depend on the weight names in `SMFNet` and `CNN` and on the
  field names of `SMFNetConfig` and `CNNConfig`. Do not rename or remove them,
  and do not change the input or output encodings, without discussing it first.
  Adding a config field with a default is fine.
- Load trained models with `load_model` (a `.ckpt` file, a `save_pretrained`
  directory or a Hugging Face Hub ID).
- Compile with `module.compile()` in the scripts, never
  `model = torch.compile(model)`: the in-place version keeps parameter names,
  so checkpoints load whether or not the model was compiled.
- Dependencies: add them to `[project]` and to `[tool.pixi]`. Packages with
  compiled code must come from conda-forge or bioconda (`[tool.pixi.dependencies]`
  or a feature), because PyPI source builds are disabled and the lock targets
  glibc 2.17. Re-lock with `pixi lock` and commit `pixi.lock`.
- Tests cover this package's behaviour, not that of its libraries. Keep them
  small.
