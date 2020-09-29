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
//! Options accepting <[glob:]value> pairs use the given values for matching file
//! names in input ZIP archives. Matches of former pairs are superseded by matches
//! of latter pairs. Omitting [glob:] by only passing a value assumes the * glob
//! pattern matching all file names whereas an empty glob pattern matches no file
//! names. An empty value disables the option for the file names matching the glob
//! pattern. Passing a single pair with an empty glob pattern and an empty value,
//! that is a colon only, disables an option with default values entirely as in
//! --recompress : whereas passing no pairs as in --recompress keeps assuming the
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
//!             are as requested according to --recompress and --align.
//!
//!     -f, --force
//!             Writes existing output ZIP archive
//!
//!     -r, --recompress <[glob:]method>...
//!             Writes files recompressed.
//!
//!             Supported methods are stored (uncompressed), deflated (most common),
//!             and bzip2 (high ratio). With no methods, files are compressed using
//!             their original methods. [default: stored]
//!
//!     -a, --align <[glob:]bytes>...
//!             Aligns uncompressed files.
//!
//!             Aligns uncompressed files in ZIP archives by padding local file
//!             headers to enable memory-mapping, SIMD instruction extensions like
//!             AVX-512, and dynamic loading of shared objects. [default: 64
//!             *.so:4096]
//!
//!     -s, --stack <[glob:]axis>...
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

use anyhow::{anyhow, Context, Result};
use clap::{crate_authors, crate_version, AppSettings, Clap};
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
/// Options accepting <[glob:]value> pairs use the given values for matching file names in input ZIP
/// archives. Matches of former pairs are superseded by matches of latter pairs. Omitting [glob:]
/// by only passing a value assumes the * glob pattern matching all file names whereas an empty glob
/// pattern matches no file names. An empty value disables the option for the file names matching
/// the glob pattern. Passing a single pair with an empty glob pattern and an empty value, that is a
/// colon only, disables an option with default values entirely as in --recompress : whereas passing
/// no pairs as in --recompress keeps assuming the default values.
#[derive(Clap, Debug)]
#[clap(
	version = crate_version!(),
	author = crate_authors!(),
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
	/// to --recompress and --align.
	#[clap(short, long, value_name = "ZIP")]
	output: Option<PathBuf>,
	/// Writes existing output ZIP archive.
	#[clap(short, long)]
	force: bool,
	/// Writes files recompressed.
	///
	/// Supported methods are stored (uncompressed), deflated (most common), and bzip2 (high ratio).
	/// With no methods, files are compressed using their original methods.
	#[clap(short, long, value_name = "[glob:]method", default_values = &["stored"])]
	recompress: Vec<String>,
	/// Aligns uncompressed files.
	///
	/// Aligns uncompressed files in ZIP archives by padding local file headers to enable
	/// memory-mapping, SIMD instruction extensions like AVX-512, and dynamic loading of shared
	/// objects.
	#[clap(short, long, value_name = "[glob:]bytes", default_values = &["64", "*.so:4096"])]
	align: Vec<String>,
	/// Stacks arrays along axis.
	///
	/// One stacked array at a time must fit twice into memory before it is written to the output
	/// ZIP archive.
	#[clap(short, long, value_name = "[glob:]axis", default_values = &["0"])]
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
				.rfind(':')
				.map(|mid| value.split_at(mid))
				.map(|(left, right)| (left, &right[1..]))
				.unwrap_or(("*", value));
			Pattern::new(left)
				.with_context(|| format!("Invalid glob pattern `{}`", left))
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
		match method {
			"stored" => Ok(CompressionMethod::Stored),
			"deflated" => Ok(CompressionMethod::Deflated),
			"bzip2" => Ok(CompressionMethod::Bzip2),
			_ => Err(anyhow!("Unsupported method `{}`", method)),
		}
		.with_context(|| format!("Invalid recompress method `{}`", method))
	})?;
	let align = parse_glob_value(&align, |bytes| {
		bytes
			.parse()
			.map_err(From::from)
			.and_then(|bytes: u16| {
				if bytes != 0 && bytes & bytes.wrapping_sub(1) == 0 {
					Ok(bytes)
				} else {
					Err(anyhow!("Must be a power of two"))
				}
			})
			.with_context(|| format!("Invalid align bytes `{}`", bytes))
	})?;
	let stack = parse_glob_value(&stack, |axis| {
		axis.parse()
			.with_context(|| format!("Invalid stack axis `{}`", axis))
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
				.with_context(|| format!("Cannot create output ZIP archive `{}`", path.display()))
		})
		.transpose()?;
	let mut zips = inputs
		.iter()
		.map(|path| {
			OpenOptions::new()
				.read(true)
				.open(path)
				.with_context(|| format!("Cannot open input ZIP archive `{}`", path.display()))
				.map(BufReader::new)
				.and_then(|zip| {
					ZipArchive::new(zip).with_context(|| {
						format!("Cannot read input ZIP archive `{}`", path.display())
					})
				})
		})
		.collect::<Result<Vec<_>>>()?;
	let mut files = IndexMap::<_, Vec<_>>::new();
	for (input, (path, zip)) in inputs.iter().zip(&mut zips).enumerate() {
		for index in 0..zip.len() {
			let file = zip.by_index(index).with_context(|| {
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
			let (is_dir, method, recompress, options) = {
				let file = files
					.last()
					.copied()
					.map(|(input, index)| zips[input].by_index(index))
					.unwrap()
					.unwrap();
				let is_dir = file.is_dir();
				let (method, recompress) = match match_glob_value(&recompress, name) {
					Some(method) => (method, file.compression() != method),
					None => (file.compression(), false),
				};
				let options = FileOptions::default()
					.compression_method(method)
					.last_modified_time(file.last_modified());
				let options = file
					.unix_mode()
					.map_or(options, |mode| options.unix_permissions(mode));
				(is_dir, method, recompress, options)
			};
			if is_dir {
				if verbose > 0 {
					println!("`{}`: add directory", name);
				}
				zip.add_directory(name, options).with_context(|| {
					format!(
						"Cannot add directory to output ZIP archive `{}`",
						path.display()
					)
				})?;
				continue;
			}
			let bytes = if method == CompressionMethod::Stored {
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
						.with_context(|| {
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
						"`{}`: start file {}-{}",
						name,
						method.to_string().to_lowercase(),
						if recompress {
							"recompressed"
						} else {
							"compressed"
						}
					);
				}
				zip.start_file(name, options).with_context(|| {
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
				let (input, file) = &mut files
					.last()
					.copied()
					.map(|(input, index)| (&inputs[input], zips[input].by_index(index).unwrap()))
					.unwrap();
				if verbose > 0 {
					println!("`{}`: merge from `{}`", name, input.display());
				}
				copy(file, zip).with_context(|| {
					format!(
						"Cannot write file to output ZIP archive `{}`",
						path.display()
					)
				})?;
			}
		}
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
				let (method, recompress) = match match_glob_value(&recompress, name) {
					Some(method) => (method, file.compression() != method),
					None => (file.compression(), false),
				};
				if recompress {
					if verbose > 0 {
						println!(
							"`{}`: not {}-compressed in `{}`",
							name,
							method.to_string().to_lowercase(),
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
							method.to_string().to_lowercase(),
							inputs[input].display()
						);
					}
				}
				let bytes = if method == CompressionMethod::Stored {
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
			(false, true) => Err(anyhow!("Not compressed but aligned as requested")),
			(true, false) => Err(anyhow!("Compressed but not aligned as requested")),
			(false, false) => Err(anyhow!("Not compressed nor aligned as requested")),
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
	if stack_npy::<f64, W, R>(path, zip, zips, files, name, axis)?.is_ok() {
		return Ok(());
	}
	if stack_npy::<f32, W, R>(path, zip, zips, files, name, axis)?.is_ok() {
		return Ok(());
	}
	if stack_npy::<i64, W, R>(path, zip, zips, files, name, axis)?.is_ok() {
		return Ok(());
	}
	if stack_npy::<u64, W, R>(path, zip, zips, files, name, axis)?.is_ok() {
		return Ok(());
	}
	if stack_npy::<i32, W, R>(path, zip, zips, files, name, axis)?.is_ok() {
		return Ok(());
	}
	if stack_npy::<u32, W, R>(path, zip, zips, files, name, axis)?.is_ok() {
		return Ok(());
	}
	if stack_npy::<i16, W, R>(path, zip, zips, files, name, axis)?.is_ok() {
		return Ok(());
	}
	if stack_npy::<u16, W, R>(path, zip, zips, files, name, axis)?.is_ok() {
		return Ok(());
	}
	if stack_npy::<i8, W, R>(path, zip, zips, files, name, axis)?.is_ok() {
		return Ok(());
	}
	if stack_npy::<u8, W, R>(path, zip, zips, files, name, axis)?.is_ok() {
		return Ok(());
	}
	stack_npy::<bool, W, R>(path, zip, zips, files, name, axis)?
}

fn stack_npy<A, W, R>(
	path: &Path,
	zip: &mut ZipWriter<W>,
	zips: &mut [ZipArchive<R>],
	files: &[(usize, usize)],
	name: &str,
	axis: usize,
) -> Result<Result<()>>
where
	A: ReadableElement + WritableElement + Copy,
	W: Write + Seek,
	R: Read + Seek,
{
	let stack = || format!("Cannot stack `{}`", name);
	let mut arrays = Vec::new();
	for (input, index) in files.iter().copied() {
		let file = zips[input].by_index(index).unwrap();
		let array = match ArrayD::<A>::read_npy(file) {
			Ok(arr) => arr,
			Err(ReadNpyError::WrongDescriptor(_)) => {
				return Ok(Err(anyhow!("Unsupported data-type")).with_context(stack))
			}
			Err(err) => return Err(err).with_context(stack),
		};
		arrays.push(array);
	}
	let arrays = arrays.iter().map(ArrayD::view).collect::<Vec<_>>();
	let array = ndarray::stack(Axis(axis), &arrays).with_context(stack)?;
	array.write_npy(zip).with_context(|| {
		format!(
			"Cannot write file to output ZIP archive `{}`",
			path.display()
		)
	})?;
	Ok(Ok(()))
}
