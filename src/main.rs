//! Merges ZIP/NPZ archives recompressed or aligned and stacks NPY arrays
//!
//! # Installation
//!
//! ```sh
//! cargo install rezip
//! ```
//!
//! # Command-line Interface
//!
//! ```text
//! Rezip 0.2.0
//! Rouven Spreckels <rs@qu1x.dev>
//! Merges ZIP/NPZ archives recompressed or aligned and stacks NPY arrays
//!
//! Options accepting <[glob=]value> pairs use the given values for matching file
//! names in input ZIP archives. Matches of former pairs are superseded by matches
//! of latter pairs. Omitting [glob=] by only passing a value assumes the * glob
//! pattern matching all file names whereas an empty glob pattern matches no file
//! names. An empty value disables the option for the file names matching the glob
//! pattern. Passing a single pair with an empty glob pattern and an empty value,
//! that is a = only, disables an option with default values entirely as in
//! --recompress = whereas passing no pairs as in --recompress keeps assuming the
//! default values.
//!
//! Usage: rezip [OPTIONS] [glob]...
//!
//! Arguments:
//!   [glob]...
//!           Merges or checks input ZIP archives.
//!           
//!           Stacks identically named files in different input ZIP archives in the
//!           order given by parsing supported file formats like NPY (NumPy array
//!           file). Otherwise, only the file in the last given input ZIP archive is
//!           merged into the output ZIP archive.
//!
//! Options:
//!   -o, --output <path>
//!           Writes output ZIP archive.
//!           
//!           With no output ZIP archive, checks if files in input ZIP archives are
//!           as requested according to --recompress and --align. Recompress levels
//!           and --merge matches are not checked.
//!
//!   -f, --force
//!           Writes existing output ZIP archive
//!
//!   -m, --merge <[glob=]name>
//!           Merges files as if they were in ZIP archives.
//!           
//!           Merges files as if they were in different ZIP archives and renames
//!           them to the given names. With empty names, keeps original names,
//!           effectively creating a ZIP archive from input files.
//!           
//!           Note: File permissions and its last modification time are not yet
//!           supported.
//!
//!   -r, --recompress <[glob=]method>
//!           Writes files recompressed.
//!           
//!           Supported methods are stored (uncompressed), deflated (most common),
//!           bzip2[:1-9] (high ratio) with 9 as default level, and zstd[:1-21]
//!           (modern) with 3 as default level. With no methods, files are
//!           recompressed using their original methods but with default levels.
//!           
//!           [default: stored]
//!
//!   -a, --align <[glob=]bytes>
//!           Aligns uncompressed files.
//!           
//!           Aligns uncompressed files in ZIP archives by padding local file
//!           headers to enable memory-mapping, SIMD instruction extensions like
//!           AVX-512, and dynamic loading of shared objects.
//!           
//!           [default: 64 *.so=4096]
//!
//!   -s, --stack <[glob=]axis>
//!           Stacks arrays along axis.
//!           
//!           One stacked array at a time must fit twice into memory before it is
//!           written to the output ZIP archive.
//!           
//!           [default: 0]
//!
//!   -v, --verbose...
//!           Prints status information.
//!           
//!           The more occurrences, the more verbose, with three at most.
//!
//!   -h, --help
//!           Print help (see a summary with '-h')
//!
//!   -V, --version
//!           Print version
//! ```

use anyhow::{Context, Result, anyhow};
use clap::{ArgAction, Parser};
use glob::{Pattern, glob as glob_expand};
use indexmap::IndexMap;
use ndarray_npz::{
    ndarray::{ArrayD, Axis, concatenate},
    ndarray_npy::{ReadNpyError, ReadNpyExt, ReadableElement, WritableElement, WriteNpyExt},
};
use std::{
    ffi::OsStr,
    fs::{self, Metadata, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Seek, Write, copy},
    path::{Path, PathBuf},
};
use walkdir::WalkDir;
use zip::{
    CompressionMethod, DateTime, ZipArchive, ZipWriter, read::ZipFile, write::SimpleFileOptions,
};

const HELP: &str = "\
{before-help}{name} {version}
{author-with-newline}{about-with-newline}
{usage-heading} {usage}

{all-args}{after-help}
";

/// Merges ZIP/NPZ archives recompressed or aligned and stacks NPY arrays
///
/// Options accepting <[glob=]value> pairs use the given values for matching file names in input ZIP
/// archives. Matches of former pairs are superseded by matches of latter pairs. Omitting [glob=]
/// by only passing a value assumes the * glob pattern matching all file names whereas an empty glob
/// pattern matches no file names. An empty value disables the option for the file names matching
/// the glob pattern. Passing a single pair with an empty glob pattern and an empty value, that is a
/// = only, disables an option with default values entirely as in --recompress = whereas passing no
/// pairs as in --recompress keeps assuming the default values.
#[derive(Parser, Debug)]
#[clap(
	name = "Rezip",
	author, version,
	propagate_version = true,
	help_template = HELP,
	arg_required_else_help = true,
)]
struct Rezip {
    /// Merges or checks input ZIP archives.
    ///
    /// Stacks identically named files in different input ZIP archives in the order given by parsing
    /// supported file formats like NPY (NumPy array file). Otherwise, only the file in the last
    /// given input ZIP archive is merged into the output ZIP archive.
    #[clap(value_name = "glob")]
    inputs: Vec<String>,
    /// Writes output ZIP archive.
    ///
    /// With no output ZIP archive, checks if files in input ZIP archives are as requested according
    /// to --recompress and --align. Recompress levels and --merge matches are not checked.
    #[clap(short, long, value_name = "path")]
    output: Option<PathBuf>,
    /// Writes existing output ZIP archive.
    #[clap(short, long)]
    force: bool,
    /// Merges files as if they were in ZIP archives.
    ///
    /// Merges files as if they were in different ZIP archives and renames them to the given names.
    /// With empty names, keeps original names, effectively creating a ZIP archive from input files.
    ///
    /// Note: File permissions and its last modification time are not yet supported.
    #[clap(short, long, value_name = "[glob=]name")]
    merge: Vec<String>,
    /// Writes files recompressed.
    ///
    /// Supported methods are stored (uncompressed), deflated (most common), bzip2[:1-9] (high
    /// ratio) with 9 as default level, and zstd[:1-21] (modern) with 3 as default level. With no
    /// methods, files are recompressed using their original methods but with default levels.
    #[clap(short, long, value_name = "[glob=]method", default_values = &["stored"])]
    recompress: Vec<String>,
    /// Aligns uncompressed files.
    ///
    /// Aligns uncompressed files in ZIP archives by padding local file headers to enable
    /// memory-mapping, SIMD instruction extensions like AVX-512, and dynamic loading of shared
    /// objects.
    #[clap(short, long, value_name = "[glob=]bytes", default_values = &["64", "*.so=4096"])]
    align: Vec<String>,
    /// Stacks arrays along axis.
    ///
    /// One stacked array at a time must fit twice into memory before it is written to the output
    /// ZIP archive.
    #[clap(short, long, value_name = "[glob=]axis", default_values = &["0"])]
    stack: Vec<String>,
    /// Prints status information.
    ///
    /// The more occurrences, the more verbose, with three at most.
    #[clap(short, long, action = ArgAction::Count)]
    verbose: u8,
}

fn parse_glob_value<F, T>(values: &[String], parse: F) -> Result<Vec<(Pattern, Option<T>)>>
where
    F: Fn(&str) -> Result<T>,
{
    values
        .iter()
        .map(|value| {
            let (left, right) = value
                .rfind('=')
                .map(|mid| value.split_at(mid))
                .map(|(left, right)| (left, &right[1..]))
                .unwrap_or(("*", value));
            Pattern::new(left)
                .with_context(|| format!("Invalid glob pattern {left:?}"))
                .and_then(|left| {
                    if right.is_empty() {
                        Ok(None)
                    } else {
                        parse(right).map(Some)
                    }
                    .map(|right| (left, right))
                })
        })
        .collect()
}

fn match_glob_value<T: Clone, P: AsRef<Path>>(
    values: &[(Pattern, Option<T>)],
    name: P,
) -> Option<T> {
    values
        .iter()
        .rev()
        .find_map(|(glob, value)| {
            if glob.matches_path(name.as_ref()) {
                Some(value)
            } else {
                None
            }
        })
        .cloned()
        .flatten()
}

enum Input<D: Read, Z: Read + Seek> {
    Dir(DirArchive<D>),
    Zip(ZipArchive<Z>),
}

struct DirArchive<D: Read> {
    files: IndexMap<usize, DirFile<D>>,
}

impl<D: Read> DirArchive<D> {
    fn len(&self) -> usize {
        self.files.len()
    }
    fn by_index(&mut self, index: usize) -> Option<&mut DirFile<D>> {
        self.files.get_mut(&index)
    }
}

struct DirFile<R: Read> {
    name: String,
    #[allow(dead_code)] // TODO
    metadata: Metadata,
    reader: Option<R>,
}

impl DirFile<BufReader<fs::File>> {
    fn new(name: String, metadata: Metadata) -> Result<Self> {
        let reader = if metadata.is_dir() {
            None
        } else {
            Some(
                OpenOptions::new()
                    .read(true)
                    .open(&name)
                    .with_context(|| format!("Cannot open input file {name:?}"))
                    .map(BufReader::new)?,
            )
        };
        Ok(DirFile {
            name,
            metadata,
            reader,
        })
    }
}

#[allow(clippy::large_enum_variant)]
enum File<'a, R: Read> {
    DirFile(&'a mut DirFile<R>),
    ZipFile(ZipFile<'a>),
}

impl<R: Read> File<'_, R> {
    fn name(&self) -> &Path {
        match self {
            Self::DirFile(file) => Path::new(&file.name),
            Self::ZipFile(file) => Path::new(file.name()),
        }
    }
    fn compression(&self) -> CompressionMethod {
        match self {
            Self::DirFile(_file) => CompressionMethod::Stored,
            Self::ZipFile(file) => file.compression(),
        }
    }
    fn last_modified(&self) -> Option<DateTime> {
        match self {
            Self::DirFile(_file) => DateTime::try_from_msdos(0, 0).ok(),
            Self::ZipFile(file) => file.last_modified(),
        }
    }
    fn is_dir(&self) -> bool {
        match self {
            Self::DirFile(file) => file.reader.is_none(),
            Self::ZipFile(file) => file.is_dir(),
        }
    }
    fn unix_mode(&self) -> Option<u32> {
        match self {
            Self::DirFile(_file) => None, // TODO
            Self::ZipFile(file) => file.unix_mode(),
        }
    }
    fn data_start(&self) -> Option<u64> {
        match self {
            Self::DirFile(_file) => None,
            Self::ZipFile(file) => Some(file.data_start()),
        }
    }
}

impl<R: Read> Read for File<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::DirFile(file) => {
                if let Some(file) = &mut file.reader {
                    file.read(buf)
                } else {
                    Err(io::Error::new(io::ErrorKind::Other, "Not readable"))
                }
            }
            Self::ZipFile(file) => file.read(buf),
        }
    }
}

impl<D: Read, Z: Read + Seek> Input<D, Z> {
    fn len(&self) -> usize {
        match self {
            Self::Dir(dir) => dir.len(),
            Self::Zip(zip) => zip.len(),
        }
    }
    fn by_index(&mut self, index: usize) -> Option<File<D>> {
        match self {
            Self::Dir(dir) => dir.by_index(index).map(File::DirFile),
            Self::Zip(zip) => zip.by_index(index).map(File::ZipFile).ok(),
        }
    }
}

impl Input<BufReader<fs::File>, BufReader<fs::File>> {
    fn new<P: AsRef<Path>>(path: P, merge: &[(Pattern, Option<String>)]) -> Result<Self> {
        let path = path.as_ref();
        let metadata =
            fs::metadata(path).with_context(|| format!("Cannot get metadata of {path:?}"))?;
        if let Some(name) = match_glob_value(merge, path) {
            let mut files = IndexMap::new();
            let file = DirFile::new(name, metadata)?;
            files.insert(0, file);
            Ok(Self::Dir(DirArchive { files }))
        } else if metadata.is_dir() {
            let mut files = IndexMap::new();
            let entries = WalkDir::new(path)
                .follow_links(true)
                .sort_by(|a, b| a.file_name().cmp(b.file_name()))
                .into_iter();
            for (index, entry) in entries.enumerate() {
                let entry = entry.with_context(|| format!("Cannot traverse {path:?}"))?;
                let name = entry
                    .path()
                    .to_str()
                    .ok_or_else(|| anyhow!("Invalid file name {:?}", entry.path()))?
                    .to_string();
                let metadata = entry
                    .metadata()
                    .with_context(|| format!("Cannot get metadata of {name:?}"))?;
                let file = DirFile::new(name, metadata)?;
                files.insert(index, file);
            }
            Ok(Self::Dir(DirArchive { files }))
        } else {
            OpenOptions::new()
                .read(true)
                .open(path)
                .with_context(|| format!("Cannot open input ZIP archive {path:?}"))
                .map(BufReader::new)
                .and_then(|zip| {
                    ZipArchive::new(zip)
                        .with_context(|| format!("Cannot read input ZIP archive {path:?}"))
                })
                .map(Self::Zip)
        }
    }
}

fn main() -> Result<()> {
    let Rezip {
        inputs,
        output,
        force,
        merge,
        recompress,
        align,
        stack,
        verbose,
    } = Rezip::parse();
    let merge = parse_glob_value(&merge, |name| Ok(name.to_string()))?;
    let recompress = parse_glob_value(&recompress, |method| {
        let mut parameters = method.split(':');
        let (algorithm, level) = (parameters.next(), parameters.next());
        match (algorithm, level) {
            (Some("stored"), None) => Ok((CompressionMethod::Stored, None)),
            (Some("deflated"), None) => Ok((CompressionMethod::Deflated, None)),
            (Some("bzip2"), level) => level
                .map_or(Ok(Some(9)), |level| {
                    level.parse::<i64>().map_err(From::from).and_then(|level| {
                        if (1..=9).contains(&level) {
                            Ok(Some(level))
                        } else {
                            Err(anyhow!("Invalid level in {method:?}"))
                        }
                    })
                })
                .map(|level| (CompressionMethod::Bzip2, level)),
            (Some("zstd"), level) => level
                .map_or(Ok(Some(3)), |level| {
                    level.parse::<i64>().map_err(From::from).and_then(|level| {
                        if (1..=21).contains(&level) {
                            Ok(Some(level))
                        } else {
                            Err(anyhow!("Invalid level in {method:?}"))
                        }
                    })
                })
                .map(|level| (CompressionMethod::Zstd, level)),
            (Some(_), _) => Err(anyhow!("Unsupported method {method:?}")),
            _ => Err(anyhow!("Invalid method {:?}", method)),
        }
        .with_context(|| format!("Invalid recompress method {method:?}"))
    })?;
    let align = parse_glob_value(&align, |bytes| {
        bytes
            .parse::<u16>()
            .map_err(From::from)
            .and_then(|bytes| {
                if bytes != 0 && bytes & bytes.wrapping_sub(1) == 0 {
                    Ok(bytes)
                } else {
                    Err(anyhow!("Must be a power of two"))
                }
            })
            .with_context(|| format!("Invalid align bytes {bytes:?}"))
    })?;
    let stack = parse_glob_value(&stack, |axis| {
        axis.parse()
            .with_context(|| format!("Invalid stack axis {axis:?}"))
    })?;
    let zip = output
        .as_ref()
        .map(|path| {
            OpenOptions::new()
                .create_new(!force)
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(path)
                .map(BufWriter::new)
                .map(ZipWriter::new)
                .with_context(|| format!("Cannot create output ZIP archive {path:?}"))
        })
        .transpose()?;
    let mut zips = Vec::new();
    let mut paths = Vec::new();
    for glob in &inputs {
        let inputs = glob_expand(glob).with_context(|| format!("Invalid glob pattern {glob:?}"))?;
        for path in inputs {
            let path = path.with_context(|| format!("Cannot read matches of {glob:?}"))?;
            let zip = Input::new(&path, &merge)?;
            paths.push(path);
            zips.push(zip);
        }
    }
    let inputs = paths;
    let files = {
        let mut files = IndexMap::<_, Vec<_>>::new();
        for (input, (path, zip)) in inputs.iter().zip(&mut zips).enumerate() {
            if verbose > 0 {
                println!(
                    "{path:?}: indexing {} file{}",
                    zip.len(),
                    if zip.len() > 1 { "s" } else { "" },
                );
            }
            for index in 0..zip.len() {
                let file = zip.by_index(index).ok_or_else(|| {
                    anyhow!("Cannot read file[{index}] in input ZIP archive {path:?}",)
                })?;
                let name = file.name().to_path_buf();
                files.entry(name).or_default().push((input, index));
            }
        }
        files
    };
    if let Some((path, mut zip)) = output.as_ref().zip(zip) {
        for (name, files) in &files {
            let extension = Path::new(&name).extension().and_then(OsStr::to_str);
            let (is_dir, algorithm, level, alignment, options) = {
                let file = files
                    .last()
                    .copied()
                    .map(|(input, index)| zips[input].by_index(index))
                    .unwrap()
                    .unwrap();
                let is_dir = file.is_dir();
                let (algorithm, level) = match match_glob_value(&recompress, name) {
                    Some((algorithm, level)) => (algorithm, level),
                    None => (file.compression(), None),
                };
                let alignment = if algorithm == CompressionMethod::Stored {
                    match_glob_value(&align, name)
                } else {
                    None
                }
                .unwrap_or_default();
                let options = SimpleFileOptions::default()
                    .compression_method(algorithm)
                    .compression_level(level)
                    .with_alignment(alignment)
                    .last_modified_time(
                        file.last_modified()
                            .ok_or_else(|| anyhow!("Missing last modified time from {path:?}"))?,
                    )
                    .large_file(true);
                let options = file
                    .unix_mode()
                    .map_or(options, |mode| options.unix_permissions(mode));
                (is_dir, algorithm, level, alignment, options)
            };
            if is_dir {
                if verbose > 0 {
                    println!("{name:?}: merging directory from {path:?}");
                }
                zip.add_directory(name.to_str().unwrap(), options)
                    .with_context(|| {
                        format!("Cannot add directory to output ZIP archive {path:?}")
                    })?;
                continue;
            }
            if alignment > 1 {
                if verbose > 0 {
                    println!("{name:?}: starting file {alignment}-byte aligned");
                }
                zip.start_file(name.to_str().unwrap(), options)
                    .with_context(|| format!("Cannot start file in output ZIP archive {path:?}"))?;
            } else {
                if verbose > 0 {
                    println!(
                        "{name:?}: starting file {}{}-recompressed",
                        algorithm.to_string().to_lowercase(),
                        level.map_or(String::new(), |level| format!(":{}", level)),
                    );
                }
                zip.start_file(name.to_str().unwrap(), options)
                    .with_context(|| format!("Cannot start file in output ZIP archive {path:?}"))?;
            }
            let stack_extensions = [Some("npy")];
            let axis = if files.len() > 1 && stack_extensions.contains(&extension) {
                match_glob_value(&stack, name)
            } else {
                None
            };
            if let Some(axis) = axis {
                if verbose > 0 {
                    println!("{name:?}: stacking {} files", files.len());
                }
                if verbose > 2 {
                    for (input, _index) in files.iter().copied() {
                        println!("{name:?}: stacking from {:?}", inputs[input]);
                    }
                }
                match extension {
                    Some("npy") => try_stack_npy(path, &mut zip, &mut zips, files, name, axis)?,
                    _ => unreachable!(),
                }
            } else {
                let (input, ref mut file) = files
                    .last()
                    .copied()
                    .map(|(input, index)| (input, zips[input].by_index(index).unwrap()))
                    .unwrap();
                if verbose > 0 {
                    println!("{name:?}: merging from {:?}", inputs[input]);
                }
                copy(file, &mut zip)
                    .with_context(|| format!("Cannot write file to output ZIP archive {path:?}"))?;
            }
        }
        if verbose > 0 {
            println!("{:?}: finishing", path);
        }
        zip.finish()
            .and_then(|mut zip| zip.flush().map_err(From::from))
            .with_context(|| format!("Cannot write file to output ZIP archive {path:?}"))?;
        Ok(())
    } else {
        let mut compressed = true;
        let mut aligned = true;
        for (name, files) in &files {
            for (input, index) in files.iter().copied() {
                let file = zips[input].by_index(index).unwrap();
                if file.is_dir() {
                    continue;
                }
                let (algorithm, _level, recompress) = match match_glob_value(&recompress, name) {
                    Some((algorithm, level)) => (algorithm, level, file.compression() != algorithm),
                    None => (file.compression(), None, false),
                };
                if recompress {
                    if verbose > 0 {
                        println!(
                            "{name:?}: not {}-compressed in {:?}",
                            algorithm.to_string().to_lowercase(),
                            inputs[input]
                        );
                    }
                    compressed = false;
                    continue;
                } else if verbose > 1 {
                    println!(
                        "{name:?}: {}-compressed in {:?}",
                        algorithm.to_string().to_lowercase(),
                        inputs[input]
                    );
                }
                let bytes = if algorithm == CompressionMethod::Stored {
                    match_glob_value(&align, name)
                } else {
                    None
                };
                if let Some((data_start, bytes)) = file.data_start().zip(bytes) {
                    if data_start % bytes as u64 == 0 {
                        if verbose > 1 {
                            println!("{name:?}: {bytes}-byte aligned in {:?}", inputs[input]);
                        }
                    } else {
                        if verbose > 0 {
                            println!("{name:?}: not {bytes}-byte aligned in {:?}", inputs[input]);
                        }
                        aligned = false;
                    }
                }
            }
        }
        match (compressed, aligned) {
            (true, true) => {
                if verbose > 0 {
                    println!("Compressed and aligned as requested");
                }
                Ok(())
            }
            (false, true) => Err(anyhow!("Not compressed but aligned as requested")),
            (true, false) => Err(anyhow!("Compressed but not aligned as requested")),
            (false, false) => Err(anyhow!("Not compressed nor aligned as requested")),
        }
    }
}

fn try_stack_npy<W, D, Z>(
    path: &Path,
    zip: &mut ZipWriter<W>,
    zips: &mut [Input<D, Z>],
    files: &[(usize, usize)],
    name: &Path,
    axis: usize,
) -> Result<()>
where
    W: Write + Seek,
    D: Read,
    Z: Read + Seek,
{
    let name = || format!("Cannot stack {:?}", name);
    if stack_npy::<f64, W, D, Z, _>(path, zip, zips, files, name, axis)? {
        return Ok(());
    }
    if stack_npy::<f32, W, D, Z, _>(path, zip, zips, files, name, axis)? {
        return Ok(());
    }
    if stack_npy::<i64, W, D, Z, _>(path, zip, zips, files, name, axis)? {
        return Ok(());
    }
    if stack_npy::<u64, W, D, Z, _>(path, zip, zips, files, name, axis)? {
        return Ok(());
    }
    if stack_npy::<i32, W, D, Z, _>(path, zip, zips, files, name, axis)? {
        return Ok(());
    }
    if stack_npy::<u32, W, D, Z, _>(path, zip, zips, files, name, axis)? {
        return Ok(());
    }
    if stack_npy::<i16, W, D, Z, _>(path, zip, zips, files, name, axis)? {
        return Ok(());
    }
    if stack_npy::<u16, W, D, Z, _>(path, zip, zips, files, name, axis)? {
        return Ok(());
    }
    if stack_npy::<i8, W, D, Z, _>(path, zip, zips, files, name, axis)? {
        return Ok(());
    }
    if stack_npy::<u8, W, D, Z, _>(path, zip, zips, files, name, axis)? {
        return Ok(());
    }
    if stack_npy::<bool, W, D, Z, _>(path, zip, zips, files, name, axis)? {
        return Ok(());
    }
    Err(anyhow!("Unsupported data-type")).with_context(name)
}

fn stack_npy<A, W, D, Z, F>(
    path: &Path,
    zip: &mut ZipWriter<W>,
    zips: &mut [Input<D, Z>],
    files: &[(usize, usize)],
    name: F,
    axis: usize,
) -> Result<bool>
where
    A: ReadableElement + WritableElement + Copy,
    W: Write + Seek,
    D: Read,
    Z: Read + Seek,
    F: Fn() -> String,
{
    let mut arrays = Vec::new();
    for (input, index) in files.iter().copied() {
        let file = zips[input].by_index(index).unwrap();
        let array = match ArrayD::<A>::read_npy(file) {
            Ok(arr) => arr,
            Err(ReadNpyError::WrongDescriptor(_)) => return Ok(false),
            Err(err) => return Err(err).with_context(name),
        };
        arrays.push(array);
    }
    let arrays = arrays.iter().map(ArrayD::view).collect::<Vec<_>>();
    let array = concatenate(Axis(axis), &arrays).with_context(name)?;
    array
        .write_npy(zip)
        .with_context(|| format!("Cannot write file to output ZIP archive {:?}", path))?;
    Ok(true)
}
