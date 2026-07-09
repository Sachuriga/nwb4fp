# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-07-09

Repository restructure to make the project publishable — a clean installable library
plus a cleanly separated paper-reproducibility section. No changes to the scientific
behaviour of the conversion pipeline.

### Added
- `pyproject.toml` (PEP 621) with curated dependencies and optional extras
  (`phy`, `dlc`, `all`, `dev`), replacing `setup.py`.
- Public API on the top-level package: `nwb4fp.__version__`, `nwb4fp.run_qmnwb`,
  `nwb4fp.test_qmnwb` (lazily imported).
- GitHub Actions CI: ruff lint, pytest (Python 3.9–3.11), build + `twine check`, and a
  guard that keeps `paper/` and `examples/` out of the wheel.
- `paper/README.md` (reproduction steps + data-availability statement) and this changelog.

### Changed
- Moved the CR–CA1 paper code out of the package to top-level `paper/`, and demo
  notebooks / run-scripts to `examples/`.
- Pointed the `analyses` test suite at the vendored `nwb4fp.analyses` namespace so it can
  collect and run (previously imported the uninstalled upstream `spatial_maps`).
- Fixed metadata: real package description and URLs, aligned `CITATION.cff`.

### Removed
- Broken/stray files: the root `__init__.py` (leftover spatial-maps re-export),
  `src/__init__.py`, `Thumbs.db`, `.code-workspace` files, loose PNGs, the `--update-deps`
  file, and the dead `postprocess/analysis/` directory.
- The UTF-16 `pip freeze` `requirements.txt` and `requirements_mac.txt` (dependencies now
  live in `pyproject.toml`).
- Deduplicated `ASSY-236-F.prb` (9 copies) down to a single packaged copy under
  `nwb4fp/data/`.
