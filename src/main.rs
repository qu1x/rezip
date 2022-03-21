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
//! rezip 0.1.2
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
//! USAGE:
//!     rezip [OPTIONS] [--] [glob]...
//!
//! ARGS:
//!     <glob>...
//!             Merges or checks input ZIP archives.
//!
//!             Stacks identically named files in different input ZIP archives in
//!             the order given by parsing supported file formats like NPY (NumPy
//!             array file). Otherwise, only the file in the last given input ZIP
//!             archive is merged into the output ZIP archive.
//!
//! OPTIONS:
//!     -o, --output <path>
//!             Writes output ZIP archive.
//!
//!             With no output ZIP archive, checks if files in input ZIP archives
//!             are as requested according to --recompress and --align. Recompress
//!             levels and --merge matches are not checked.
//!
//!     -f, --force
//!             Writes existing output ZIP archive
//!
//!     -m, --merge <[glob=]name>...
//!             Merges files as if they were in ZIP archives.
//!
//!             Merges files as if they were in different ZIP archives and renames
//!             them to the given names. With empty names, keeps original names,
//!             effectively creating a ZIP archive from input files.
//!
//!             Note: File permissions and its last modification time are not yet
//!             supported.
//!
//!     -r, --recompress <[glob=]method>...
//!             Writes files recompressed.
//!
//!             Supported methods are stored (uncompressed), deflated (most common),
//!             bzip2[:1-9] (high ratio) with 9 as default level, and zstd[:1-21]
//!             (modern) with 3 as default level. With no methods, files are
//!             recompressed using their original methods but with default levels.
//!
//!             Note: Compression levels and method zstd are not yet supported.
//!             [default: stored]
//!
//!     -a, --align <[glob=]bytes>...
//!             Aligns uncompressed files.
//!
//!             Aligns uncompressed files in ZIP archives by padding local file
//!             headers to enable memory-mapping, SIMD instruction extensions like
//!             AVX-512, and dynamic loading of shared objects. [default: 64
//!             *.so=4096]
//!
//!     -s, --stack <[glob=]axis>...
//!             Stacks arrays along axis.
//!
//!             One stacked array at a time must fit twice into memory before it is
//!             written to the output ZIP archive. [default: 0]
//!
//!     -v, --verbose
//!             Prints status information.
//!
//!             The more occurrences, the more verbose, with three at most.
//!
//!     -h, --help
//!             Prints help information
//!
//!     -V, --version
//!             Prints version information
//!
//! ```

#![forbid(unsafe_code)]
#![forbid(missing_docs)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::redundant_else)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::large_enum_variant)]

use clap::{crate_authors, crate_version, AppSettings, Parser};
use color_eyre::{eyre::eyre, eyre::WrapErr, Result};
use glob::{glob as glob_expand, Pattern};
use indexmap::IndexMap;
use ndarray::{ArrayD, Axis};
use ndarray_npy::{ReadNpyError, ReadNpyExt, ReadableElement, WritableElement, WriteNpyExt};
use std::ffi::OsStr;
use std::fs::{self, Metadata, OpenOptions};
use std::io::{self, copy, BufReader, BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use zip::{read::ZipFile, write::FileOptions, CompressionMethod, DateTime, ZipArchive, ZipWriter};

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
	version = crate_version!(),
	author = crate_authors!(),
	global_setting = AppSettings::DeriveDisplayOrder,
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
	///
	/// Note: Compression levels and method zstd are not yet supported.
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
	#[clap(short, long, parse(from_occurrences))]
	verbose: u64,
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
				.wrap_err_with(|| format!("Invalid glob pattern {:?}", left))
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
					.wrap_err_with(|| format!("Cannot open input file {:?}", name))
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

enum File<'a, R: Read> {
	DirFile(&'a mut DirFile<R>),
	ZipFile(ZipFile<'a>),
}

impl<'a, R: Read> File<'a, R> {
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
	fn last_modified(&self) -> DateTime {
		match self {
			Self::DirFile(_file) => DateTime::from_msdos(0, 0), // TODO
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

impl<'a, R: Read> Read for File<'a, R> {
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
			fs::metadata(path).wrap_err_with(|| format!("Cannot get metadata of {:?}", path))?;
		if let Some(name) = match_glob_value(merge, path) {
			let mut files = IndexMap::new();
			let file = DirFile::new(name, metadata)?;
			files.insert(0, file);
			Ok(Self::Dir(DirArchive { files }))
		} else {
			if metadata.is_dir() {
				let mut files = IndexMap::new();
				let entries = WalkDir::new(path)
					.follow_links(true)
					.sort_by(|a, b| a.file_name().cmp(b.file_name()))
					.into_iter();
				for (index, entry) in entries.enumerate() {
					let entry = entry.wrap_err_with(|| format!("Cannot traverse {:?}", path))?;
					let name = entry
						.path()
						.to_str()
						.ok_or_else(|| eyre!("Invalid file name {:?}", entry.path()))?
						.to_string();
					let metadata = entry
						.metadata()
						.wrap_err_with(|| format!("Cannot get metadata of {:?}", name))?;
					let file = DirFile::new(name, metadata)?;
					files.insert(index, file);
				}
				Ok(Self::Dir(DirArchive { files }))
			} else {
				OpenOptions::new()
					.read(true)
					.open(&path)
					.wrap_err_with(|| format!("Cannot open input ZIP archive {:?}", path))
					.map(BufReader::new)
					.and_then(|zip| {
						ZipArchive::new(zip)
							.wrap_err_with(|| format!("Cannot read input ZIP archive {:?}", path))
					})
					.map(Self::Zip)
			}
		}
	}
}

fn main() -> Result<()> {
	color_eyre::install()?;
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
					level.parse::<i32>().map_err(From::from).and_then(|level| {
						if (1..=9).contains(&level) {
							Ok(Some(level))
						} else {
							Err(eyre!("Invalid level in {:?}", method))
						}
					})
				})
				.map(|level| (CompressionMethod::Bzip2, level)),
			// TODO
			//(Some("zstd"), level) => level
			//	.map_or(Ok(Some(3)), |level| {
			//		level.parse::<i32>().map_err(From::from).and_then(|level| {
			//			if (1..=21).contains(&level) {
			//				Ok(Some(level))
			//			} else {
			//				Err(eyre!("Invalid level in {:?}", method))
			//			}
			//		})
			//	})
			//	.map(|level| (CompressionMethod::Zstd, level)),
			(Some(_), _) => Err(eyre!("Unsupported method {:?}", method)),
			_ => Err(eyre!("Invalid method {:?}", method)),
		}
		.wrap_err_with(|| format!("Invalid recompress method {:?}", method))
	})?;
	let align = parse_glob_value(&align, |bytes| {
		bytes
			.parse::<u16>()
			.map_err(From::from)
			.and_then(|bytes| {
				if bytes != 0 && bytes & bytes.wrapping_sub(1) == 0 {
					Ok(bytes)
				} else {
					Err(eyre!("Must be a power of two"))
				}
			})
			.wrap_err_with(|| format!("Invalid align bytes {:?}", bytes))
	})?;
	let stack = parse_glob_value(&stack, |axis| {
		axis.parse()
			.wrap_err_with(|| format!("Invalid stack axis {:?}", axis))
	})?;
	let mut zip = output
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
				.wrap_err_with(|| format!("Cannot create output ZIP archive {:?}", path))
		})
		.transpose()?;
	let mut zips = Vec::new();
	let mut paths = Vec::new();
	for glob in &inputs {
		let inputs =
			glob_expand(glob).wrap_err_with(|| format!("Invalid glob pattern {:?}", glob))?;
		for path in inputs {
			let path = path.wrap_err_with(|| format!("Cannot read matches of {:?}", glob))?;
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
					"{:?}: indexing {} file{}",
					path,
					zip.len(),
					if zip.len() > 1 { "s" } else { "" },
				);
			}
			for index in 0..zip.len() {
				let file = zip.by_index(index).ok_or_else(|| {
					eyre!(
						"Cannot read file[{}] in input ZIP archive {:?}",
						index,
						path
					)
				})?;
				let name = file.name().to_path_buf();
				files.entry(name).or_default().push((input, index));
			}
		}
		files
	};
	if let Some((path, zip)) = output.as_ref().zip(zip.as_mut()) {
		let mut total_pad_length = 0;
		for (name, files) in &files {
			let extension = Path::new(&name).extension().and_then(OsStr::to_str);
			let (is_dir, algorithm, level, options) = {
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
				let options = FileOptions::default()
					.compression_method(algorithm)
					.last_modified_time(file.last_modified())
					.large_file(true);
				// TODO
				//let options = level.map_or(options, |level| options.compression_level(level));
				let options = file
					.unix_mode()
					.map_or(options, |mode| options.unix_permissions(mode));
				(is_dir, algorithm, level, options)
			};
			if is_dir {
				if verbose > 0 {
					println!("{:?}: merging directory from {:?}", name, path);
				}
				zip.add_directory(name.to_str().unwrap(), options)
					.wrap_err_with(|| {
						format!("Cannot add directory to output ZIP archive {:?}", path)
					})?;
				continue;
			}
			let bytes = if algorithm == CompressionMethod::Stored {
				match_glob_value(&align, name)
			} else {
				None
			};
			if let Some(bytes) = bytes {
				if verbose > 0 {
					println!("{:?}: starting file {}-byte aligned", name, bytes);
				}
				let pad_length = zip
					.start_file_aligned(name.to_str().unwrap(), options, bytes)
					.wrap_err_with(|| {
						format!("Cannot start file in output ZIP archive {:?}", path)
					})?;
				if verbose > 1 {
					println!("{:?}: via {}-byte pad", name, pad_length);
				}
				total_pad_length += pad_length;
			} else {
				if verbose > 0 {
					println!(
						"{:?}: starting file {}{}-recompressed",
						name,
						algorithm.to_string().to_lowercase(),
						level.map_or(String::new(), |level| format!(":{}", level)),
					);
				}
				zip.start_file(name.to_str().unwrap(), options)
					.wrap_err_with(|| {
						format!("Cannot start file in output ZIP archive {:?}", path)
					})?;
			}
			let stack_extensions = [Some("npy")];
			let axis = if files.len() > 1 && stack_extensions.contains(&extension) {
				match_glob_value(&stack, name)
			} else {
				None
			};
			if let Some(axis) = axis {
				if verbose > 0 {
					println!("{:?}: stacking {} files", name, files.len());
				}
				if verbose > 2 {
					for (input, _index) in files.iter().copied() {
						println!("{:?}: stacking from {:?}", name, inputs[input]);
					}
				}
				match extension {
					Some("npy") => try_stack_npy(path, zip, &mut zips, &files, name, axis)?,
					_ => unreachable!(),
				}
			} else {
				let (input, ref mut file) = files
					.last()
					.copied()
					.map(|(input, index)| (input, zips[input].by_index(index).unwrap()))
					.unwrap();
				if verbose > 0 {
					println!("{:?}: merging from {:?}", name, inputs[input]);
				}
				copy(file, zip).wrap_err_with(|| {
					format!("Cannot write file to output ZIP archive {:?}", path)
				})?;
			}
		}
		if verbose > 0 {
			println!("{:?}: finishing", path);
		}
		zip.finish()
			.and_then(|mut zip| zip.flush().map_err(From::from))
			.wrap_err_with(|| format!("Cannot write file to output ZIP archive {:?}", path))?;
		if verbose > 1 {
			println!("{:?}: via {}-byte pad in total", path, total_pad_length);
		}
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
							"{:?}: not {}-compressed in {:?}",
							name,
							algorithm.to_string().to_lowercase(),
							inputs[input]
						);
					}
					compressed = false;
					continue;
				} else {
					if verbose > 1 {
						println!(
							"{:?}: {}-compressed in {:?}",
							name,
							algorithm.to_string().to_lowercase(),
							inputs[input]
						);
					}
				}
				let bytes = if algorithm == CompressionMethod::Stored {
					match_glob_value(&align, name)
				} else {
					None
				};
				if let Some((data_start, bytes)) = file.data_start().zip(bytes) {
					if data_start % bytes as u64 == 0 {
						if verbose > 1 {
							println!("{:?}: {}-byte aligned in {:?}", name, bytes, inputs[input]);
						}
					} else {
						if verbose > 0 {
							println!(
								"{:?}: not {}-byte aligned in {:?}",
								name, bytes, inputs[input]
							);
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
			(false, true) => Err(eyre!("Not compressed but aligned as requested")),
			(true, false) => Err(eyre!("Compressed but not aligned as requested")),
			(false, false) => Err(eyre!("Not compressed nor aligned as requested")),
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
	Err(eyre!("Unsupported data-type")).wrap_err_with(name)
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
			Err(err) => return Err(err).wrap_err_with(name),
		};
		arrays.push(array);
	}
	let arrays = arrays.iter().map(ArrayD::view).collect::<Vec<_>>();
	let array = ndarray::concatenate(Axis(axis), &arrays).wrap_err_with(name)?;
	array
		.write_npy(zip)
		.wrap_err_with(|| format!("Cannot write file to output ZIP archive {:?}", path))?;
	Ok(true)
}
