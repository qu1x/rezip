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
//! rezip 0.1.0
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
//!     rezip [OPTIONS] [--] [ZIP]...
//!
//! ARGS:
//!     <ZIP>...
//!             Merges or checks input ZIP archives.
//!
//!             Stacks identically named files in different input ZIP archives in
//!             the order given by parsing supported file formats like NPY (NumPy
//!             array file). Otherwise, only the file in the last given input ZIP
//!             archive is merged into the output ZIP archive.
//!
//! OPTIONS:
//!     -o, --output <ZIP>
//!             Writes output ZIP archive.
//!
//!             With no output ZIP archive, checks if files in input ZIP archives
//!             are as requested according to --recompress and --align. Recompress
//!             levels are not considered.
//!
//!     -f, --force
//!             Writes existing output ZIP archive
//!
//!     -r, --recompress <[glob=]method>...
//!             Writes files recompressed.
//!
//!             Supported methods are stored (uncompressed), deflated (most common),
//!             bzip2[:1-9] (high ratio) with 9 as default level, and zstd[:1-21]
//!             (modern) with 3 as default level. With no methods, files are
//!             recompressed using their original methods but with default levels.
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

use clap::{crate_authors, crate_version, AppSettings, Clap};
use color_eyre::{eyre::eyre, eyre::WrapErr, Result};
use glob::Pattern;
use indexmap::IndexMap;
use ndarray::{ArrayD, Axis};
use ndarray_npy::{ReadNpyError, ReadNpyExt, ReadableElement, WritableElement, WriteNpyExt};
use std::ffi::OsStr;
use std::fs::OpenOptions;
use std::io::{copy, BufReader, BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};
use zip::{write::FileOptions, CompressionMethod, ZipArchive, ZipWriter};

/// Merges ZIP/NPZ archives recompressed or aligned and stacks NPY arrays
///
/// Options accepting <[glob=]value> pairs use the given values for matching file names in input ZIP
/// archives. Matches of former pairs are superseded by matches of latter pairs. Omitting [glob=]
/// by only passing a value assumes the * glob pattern matching all file names whereas an empty glob
/// pattern matches no file names. An empty value disables the option for the file names matching
/// the glob pattern. Passing a single pair with an empty glob pattern and an empty value, that is a
/// = only, disables an option with default values entirely as in --recompress = whereas passing no
/// pairs as in --recompress keeps assuming the default values.
#[derive(Clap, Debug)]
#[clap(
	version = crate_version!(),
	author = crate_authors!(),
	max_term_width = 80,
	global_setting = AppSettings::ColoredHelp,
	global_setting = AppSettings::DeriveDisplayOrder,
	global_setting = AppSettings::UnifiedHelpMessage,
	global_setting = AppSettings::ArgRequiredElseHelp,
)]
struct Rezip {
	/// Merges or checks input ZIP archives.
	///
	/// Stacks identically named files in different input ZIP archives in the order given by parsing
	/// supported file formats like NPY (NumPy array file). Otherwise, only the file in the last
	/// given input ZIP archive is merged into the output ZIP archive.
	#[clap(value_name = "ZIP")]
	inputs: Vec<PathBuf>,
	/// Writes output ZIP archive.
	///
	/// With no output ZIP archive, checks if files in input ZIP archives are as requested according
	/// to --recompress and --align. Recompress levels are not considered.
	#[clap(short, long, value_name = "ZIP")]
	output: Option<PathBuf>,
	/// Writes existing output ZIP archive.
	#[clap(short, long)]
	force: bool,
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
				.wrap_err_with(|| format!("Invalid glob pattern `{}`", left))
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

fn match_glob_value<T: Copy>(values: &[(Pattern, Option<T>)], name: &str) -> Option<T> {
	values
		.iter()
		.rev()
		.find_map(|(glob, value)| {
			if glob.matches(name) {
				Some(value)
			} else {
				None
			}
		})
		.copied()
		.flatten()
}

fn main() -> Result<()> {
	color_eyre::install()?;
	let Rezip {
		inputs,
		output,
		force,
		recompress,
		align,
		stack,
		verbose,
	} = Rezip::parse();
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
							Err(eyre!("Invalid level in `{}`", method))
						}
					})
				})
				.map(|level| (CompressionMethod::Bzip2, level)),
			(Some("zstd"), level) => level
				.map_or(Ok(Some(3)), |level| {
					level.parse::<i32>().map_err(From::from).and_then(|level| {
						if (1..=21).contains(&level) {
							Ok(Some(level))
						} else {
							Err(eyre!("Invalid level in `{}`", method))
						}
					})
				})
				.map(|level| (CompressionMethod::Zstd, level)),
			(Some(_), _) => Err(eyre!("Unsupported method `{}`", method)),
			_ => Err(eyre!("Invalid method `{}`", method)),
		}
		.wrap_err_with(|| format!("Invalid recompress method `{}`", method))
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
			.wrap_err_with(|| format!("Invalid align bytes `{}`", bytes))
	})?;
	let stack = parse_glob_value(&stack, |axis| {
		axis.parse()
			.wrap_err_with(|| format!("Invalid stack axis `{}`", axis))
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
				.wrap_err_with(|| format!("Cannot create output ZIP archive `{}`", path.display()))
		})
		.transpose()?;
	let mut zips = inputs
		.iter()
		.map(|path| {
			OpenOptions::new()
				.read(true)
				.open(path)
				.wrap_err_with(|| format!("Cannot open input ZIP archive `{}`", path.display()))
				.map(BufReader::new)
				.and_then(|zip| {
					ZipArchive::new(zip).wrap_err_with(|| {
						format!("Cannot read input ZIP archive `{}`", path.display())
					})
				})
		})
		.collect::<Result<Vec<_>>>()?;
	let mut files = IndexMap::<_, Vec<_>>::new();
	for (input, (path, zip)) in inputs.iter().zip(&mut zips).enumerate() {
		for index in 0..zip.len() {
			let file = zip.by_index(index).wrap_err_with(|| {
				format!(
					"Cannot read file[{}] in input ZIP archive `{}`",
					index,
					path.display()
				)
			})?;
			let name = file.name().to_string();
			files.entry(name).or_default().push((input, index));
		}
	}
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
				let options = level.map_or(options, |level| options.compression_level(level));
				let options = file
					.unix_mode()
					.map_or(options, |mode| options.unix_permissions(mode));
				(is_dir, algorithm, level, options)
			};
			if is_dir {
				if verbose > 0 {
					println!("`{}`: add directory", name);
				}
				zip.add_directory(name, options).wrap_err_with(|| {
					format!(
						"Cannot add directory to output ZIP archive `{}`",
						path.display()
					)
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
					println!("`{}`: start file {}-byte aligned", name, bytes);
				}
				let pad_length =
					zip.start_file_aligned(name, options, bytes)
						.wrap_err_with(|| {
							format!(
								"Cannot start file in output ZIP archive `{}`",
								path.display()
							)
						})?;
				if verbose > 1 {
					println!("`{}`: via {}-byte pad", name, pad_length);
				}
				total_pad_length += pad_length;
			} else {
				if verbose > 0 {
					println!(
						"`{}`: start file {}{}-recompressed",
						name,
						algorithm.to_string().to_lowercase(),
						level.map_or(String::new(), |level| format!(":{}", level)),
					);
				}
				zip.start_file(name, options).wrap_err_with(|| {
					format!(
						"Cannot start file in output ZIP archive `{}`",
						path.display()
					)
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
					println!("`{}`: stack {} files", name, files.len());
				}
				if verbose > 2 {
					for (input, _index) in files.iter().copied() {
						println!("`{}`: stack from `{}`", name, inputs[input].display());
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
					println!("`{}`: merge from `{}`", name, inputs[input].display());
				}
				copy(file, zip).wrap_err_with(|| {
					format!(
						"Cannot write file to output ZIP archive `{}`",
						path.display()
					)
				})?;
			}
		}
		if verbose > 0 {
			println!("Finish `{}`", path.display());
		}
		zip.finish()
			.and_then(|mut zip| zip.flush().map_err(From::from))
			.wrap_err_with(|| {
				format!(
					"Cannot write file to output ZIP archive `{}`",
					path.display()
				)
			})?;
		if verbose > 1 {
			println!("Padded {} bytes in total", total_pad_length);
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
							"`{}`: not {}-compressed in `{}`",
							name,
							algorithm.to_string().to_lowercase(),
							inputs[input].display()
						);
					}
					compressed = false;
					continue;
				} else {
					if verbose > 1 {
						println!(
							"`{}`: {}-compressed in `{}`",
							name,
							algorithm.to_string().to_lowercase(),
							inputs[input].display()
						);
					}
				}
				let bytes = if algorithm == CompressionMethod::Stored {
					match_glob_value(&align, name)
				} else {
					None
				};
				if let Some(bytes) = bytes {
					if file.data_start() % bytes as u64 == 0 {
						if verbose > 1 {
							println!(
								"`{}`: {}-byte aligned in `{}`",
								name,
								bytes,
								inputs[input].display()
							);
						}
					} else {
						if verbose > 0 {
							println!(
								"`{}`: not {}-byte aligned in `{}`",
								name,
								bytes,
								inputs[input].display()
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

fn try_stack_npy<W, R>(
	path: &Path,
	zip: &mut ZipWriter<W>,
	zips: &mut [ZipArchive<R>],
	files: &[(usize, usize)],
	name: &str,
	axis: usize,
) -> Result<()>
where
	W: Write + Seek,
	R: Read + Seek,
{
	let name = || format!("Cannot stack `{}`", name);
	if stack_npy::<f64, W, R, _>(path, zip, zips, files, name, axis)? {
		return Ok(());
	}
	if stack_npy::<f32, W, R, _>(path, zip, zips, files, name, axis)? {
		return Ok(());
	}
	if stack_npy::<i64, W, R, _>(path, zip, zips, files, name, axis)? {
		return Ok(());
	}
	if stack_npy::<u64, W, R, _>(path, zip, zips, files, name, axis)? {
		return Ok(());
	}
	if stack_npy::<i32, W, R, _>(path, zip, zips, files, name, axis)? {
		return Ok(());
	}
	if stack_npy::<u32, W, R, _>(path, zip, zips, files, name, axis)? {
		return Ok(());
	}
	if stack_npy::<i16, W, R, _>(path, zip, zips, files, name, axis)? {
		return Ok(());
	}
	if stack_npy::<u16, W, R, _>(path, zip, zips, files, name, axis)? {
		return Ok(());
	}
	if stack_npy::<i8, W, R, _>(path, zip, zips, files, name, axis)? {
		return Ok(());
	}
	if stack_npy::<u8, W, R, _>(path, zip, zips, files, name, axis)? {
		return Ok(());
	}
	if stack_npy::<bool, W, R, _>(path, zip, zips, files, name, axis)? {
		return Ok(());
	}
	Err(eyre!("Unsupported data-type")).wrap_err_with(name)
}

fn stack_npy<A, W, R, F>(
	path: &Path,
	zip: &mut ZipWriter<W>,
	zips: &mut [ZipArchive<R>],
	files: &[(usize, usize)],
	name: F,
	axis: usize,
) -> Result<bool>
where
	A: ReadableElement + WritableElement + Copy,
	W: Write + Seek,
	R: Read + Seek,
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
	let array = ndarray::stack(Axis(axis), &arrays).wrap_err_with(name)?;
	array.write_npy(zip).wrap_err_with(|| {
		format!(
			"Cannot write file to output ZIP archive `{}`",
			path.display()
		)
	})?;
	Ok(true)
}
