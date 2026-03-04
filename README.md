# dc-tools

![Logo du PPR Océan & Climat](docs/source/_static/Logo_PPR.jpg)

This repository contains most of the codebase for the Ocean Data Challenges developed as part of the [PPR Océan & Climat](https://www.ocean-climat.fr/).

There are separate repositories for each Data Challenge, which contain their specific configuration.
All those repositories as well as additional code are hosted in the [`ocean-ai-data-challenges` GitHub organization](https://github.com/ocean-ai-data-challenges).

The code for calculating metrics is based on [Mercator Ocean International's `oceanbench` library](https://github.com/mercator-ocean/oceanbench), which we have forked.
We intend on merging back into the original `oceanbench` repository in the near future.

## Test profiles

The test suite is organized around three pytest markers:

- `unit`: fast, deterministic tests (default PR profile)
- `integration`: heavier tests involving multiple components (local I/O, Dask, external libs)
- `slow`: long-running/integration-style tests (network, large I/O, cluster-heavy)

By default, project runs exclude `slow` tests.

### Commands

- Fast PR profile (lint + types + tests):
	- `poetry run poe all`
- Slow profile (to run manually or in nightly CI):
	- `poetry run poe test-slow`
- Full profile including slow tests:
	- `poetry run poe all-with-slow`
- Strict full profile (target coverage gate):
	- `poetry run poe all-strict`

## Coverage strategy (incremental)

Current global coverage is below the long-term target. To avoid blocking productive PRs while still improving quality, use an incremental ramp:

1. Keep the fast PR gate on focused tests (`test-fast`, baseline `--cov-fail-under=20`) and track trend in CI artifacts.
2. Add coverage first on stable modules touched by active work (`dctools/data/datasets`, `dctools/metrics`, `dctools/data/connection`).
3. Raise the fast threshold in small steps once sustained for several runs (for example +5 points per cycle).
4. Run `slow` profile in nightly CI and use failures as backlog input, not immediate PR blockers.
