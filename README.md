[![Build Status][]](https://travis-ci.org/qu1x/rezip)
[![Downloads][]](https://crates.io/crates/rezip)
[![Rust][]](https://www.rust-lang.org)
[![Version][]](https://crates.io/crates/rezip)
[![License][]](https://opensource.org/licenses)

[Build Status]: https://travis-ci.org/qu1x/rezip.svg
[Downloads]: https://img.shields.io/crates/d/rezip.svg
[Rust]: https://img.shields.io/badge/rust-stable-brightgreen.svg
[Version]: https://img.shields.io/crates/v/rezip.svg
[License]: https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg

# rezip

Merges ZIP/NPZ archives recompressed or aligned and stacks NPY arrays

## Installation

```sh
cargo install rezip
```

## Command-line Interface

```text
rezip 0.1.0
Rouven Spreckels <rs@qu1x.dev>
Merges ZIP/NPZ archives recompressed or aligned and stacks NPY arrays

Options accepting <[glob=]value> pairs use the given values for matching file
names in input ZIP archives. Matches of former pairs are superseded by matches
of latter pairs. Omitting [glob=] by only passing a value assumes the * glob
pattern matching all file names whereas an empty glob pattern matches no file
names. An empty value disables the option for the file names matching the glob
pattern. Passing a single pair with an empty glob pattern and an empty value,
that is a = only, disables an option with default values entirely as in
--recompress = whereas passing no pairs as in --recompress keeps assuming the
default values.

USAGE:
    rezip [OPTIONS] [--] [glob]...

ARGS:
    <glob>...
            Merges or checks input ZIP archives.

            Stacks identically named files in different input ZIP archives in
            the order given by parsing supported file formats like NPY (NumPy
            array file). Otherwise, only the file in the last given input ZIP
            archive is merged into the output ZIP archive.

OPTIONS:
    -o, --output <path>
            Writes output ZIP archive.

            With no output ZIP archive, checks if files in input ZIP archives
            are as requested according to --recompress and --align. Recompress
            levels and --merge matches are not checked.

    -f, --force
            Writes existing output ZIP archive

    -m, --merge <[glob=]name>...
            Merges files as if they were in ZIP archives.

            Merges files as if they were in different ZIP archives and renames
            them to the given names. With empty names, keeps original names,
            effectively creating a ZIP archive from input files.

            Note: File permissions and its last modification time are not yet
            supported.

    -r, --recompress <[glob=]method>...
            Writes files recompressed.

            Supported methods are stored (uncompressed), deflated (most common),
            bzip2[:1-9] (high ratio) with 9 as default level, and zstd[:1-21]
            (modern) with 3 as default level. With no methods, files are
            recompressed using their original methods but with default levels.

            Note: Compression levels and method zstd are not yet supported.
            [default: stored]

    -a, --align <[glob=]bytes>...
            Aligns uncompressed files.

            Aligns uncompressed files in ZIP archives by padding local file
            headers to enable memory-mapping, SIMD instruction extensions like
            AVX-512, and dynamic loading of shared objects. [default: 64
            *.so=4096]

    -s, --stack <[glob=]axis>...
            Stacks arrays along axis.

            One stacked array at a time must fit twice into memory before it is
            written to the output ZIP archive. [default: 0]

    -v, --verbose
            Prints status information.

            The more occurrences, the more verbose, with three at most.

    -h, --help
            Prints help information

    -V, --version
            Prints version information

```

## License

Dual-licensed under `MIT OR Apache-2.0`.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the works by you shall be licensed as above, without any
additional terms or conditions.

## Releases

### Version 0.1.0 (2021-06-05)

  * Implement.

### Version 0.0.0 (2020-09-29)

  * Reserve name.
