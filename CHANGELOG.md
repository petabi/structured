# Changelog

This file documents recent notable changes to this project. The format of this
file is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Generated enum maps during parsing events. The enum value of each column starts with `1_u32`.
- Limited enum dimensions by a user configuration, a predefined coverage of data, and a maximum dimension.

### Changed

- `Description` contains enum fields as `String` which is converted from `u32` at `describe` functions. 
