# Changelog

This file documents recent notable changes to this project. The format of this
file is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Change tuple with struct because the openapi doesn't support tuple.
- `Description` is decomposed into `Description` and `NLargestCount`.
- Change `table::describe` with `table::statistics` which returns
  a vector of `ColumnStatistics` having `Description` and `NLargestCount`.
- Take time intervals and numbers of top N as arguments of the `statistics`
  function.

## [0.1.1] - 2020-02-25

### Added

- Interface to generate empty batch to `Reader`.

## [0.1.0] - 2020-02-13

### Added

- DataFrame-like data structure (`Table`) to represent structured data in a
  column-oriented form.
- Interface to read CSV data into `Table`.

[Unreleased]: https://github.com/petabi/structured/compare/0.1.1...master
[0.1.1]: https://github.com/petabi/structured/tree/0.1.1...0.1.0
[0.1.0]: https://github.com/petabi/structured/tree/0.1.0
