# Changelog

This file documents recent notable changes to this project. The format of this
file is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2020-06-21

### Changed

- Requires Rust 1.52 or later.
- No longer supports dimension limitation for `Enum` in `CSV`.
- `Enum` field is converted into utf8 string instead of being further processd.

## [0.4.0] - 2021-03-07

### Changed

- Requires Rust 1.46 or later.
- Replaced `array::Array` with `Array` from Apache Arrow.

## [0.3.0] - 2021-02-18

### Changed
- Requires Rust 1.42 or later.
- Requires dashmap 4 or later.

## [0.2.1] - 2020-08-26

### Added

- `Table::count_group_by` gets `by_column`, `by_interval` and `count_columns`
  as arguments for generating series having counts of `count_columns` values
  in rows grouped by `by_column` with `by_interval`.
- The series have values which sum each column of `count_columns`, so
  the number of generated series is the same number of `count_columns`.
- If one of count columns is the same as `by_column`, the series having
  the number of rows as values will be also generated and its `count_index`
  in the result will be `None`.

## [0.2.0] - 2020-03-13

### Changed

- Mitigate lock contention in parsing dictionary-encoded fields.
- `Description` is decomposed into `Description` and `NLargestCount`.
- Change `Table::describe` with `Table::statistics` which returns
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

[0.5.0]: https://github.com/petabi/structured/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/petabi/structured/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/petabi/structured/compare/0.2.1...0.3.0
[0.2.1]: https://github.com/petabi/structured/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/petabi/structured/compare/0.1.1...0.2.0
[0.1.1]: https://github.com/petabi/structured/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/petabi/structured/tree/0.1.0
