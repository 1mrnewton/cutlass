//! The stream configuration the encoder writes: block sizes, generated
//! books, a floor 1 per block size, a residue per block size and format,
//! and a mapping per block size that couples channel pairs.
//!
//! Every book is generated here from a model; nothing is a trained table.

use crate::codebook::BookSpec;
use crate::encoder::books::{Coder, Lattice, MAX_LENGTH, geometric, huffman_lengths};
use crate::header::Identification;
use crate::setup::{Floor, Floor1, Floor1Class, Mapping, Mode, Residue, SetupSpec};
use cutlass_decode::CodecError;

/// Floor 1 Y multiplier: 128 levels of about 1.09 dB.
pub(crate) const FLOOR_MULTIPLIER: u8 = 2;
pub(crate) const FLOOR_RANGE: i32 = 128;
/// Floor points per partition.
const FLOOR_DIMENSIONS: usize = 3;
/// Interior floor points for a long block of 1024 bins.
const LONG_POSTS: [u16; 30] = [
    2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224,
    256, 320, 384, 448, 512, 640, 768,
];
/// Interior floor points for a short block of 128 bins.
const SHORT_POSTS: [u16; 12] = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96];
/// The largest residue magnitude the escape cascade reaches.
pub(crate) const MAX_RESIDUE: i32 = 16 * 1089 + 16 * 33 + 16;

/// One floor: its points and the books that code them.
#[derive(Clone, Debug)]
pub(crate) struct FloorPlan {
    pub floor: Floor1,
    pub masterbook: usize,
    /// Per subclass: the book, or none (only the value 0).
    pub subbooks: [Option<usize>; 4],
    /// Per subclass: the values it codes are below this.
    pub limits: [i32; 4],
}

/// One residue classification: the bounds it codes and its passes.
#[derive(Clone, Debug)]
pub(crate) struct ClassPlan {
    /// Largest magnitude at the even and the odd positions of a vector
    /// (in format 2, the magnitude and the angle channel).
    pub limits: [i32; 2],
    /// Book and lattice per pass, coarsest first.
    pub passes: Vec<(usize, Lattice)>,
}

#[derive(Clone, Debug)]
pub(crate) struct ResiduePlan {
    pub kind: u8,
    pub begin: usize,
    pub end: usize,
    pub partition: usize,
    pub classes: Vec<ClassPlan>,
    pub classbook: usize,
    /// Classifications per classbook codeword.
    pub per_word: usize,
}

/// Channels sharing one submap: a coupled pair (format 2) or the rest
/// (format 1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Group {
    pub channels: Vec<usize>,
    pub coupled: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct Profile {
    pub channels: usize,
    pub rate: u32,
    pub blocksizes: [usize; 2],
    pub coders: Vec<Coder>,
    /// Per block size.
    pub floors: [FloorPlan; 2],
    /// Per block size: format 1, then format 2.
    pub residues: [[ResiduePlan; 2]; 2],
    pub groups: Vec<Group>,
    /// Coded bins per block size; the rest are zero.
    pub bandwidth: [usize; 2],
    pub setup: SetupSpec,
}

/// The coupled pairs for `channels` in the Vorbis order (section 4.3.9):
/// front left and right, then the surround pairs. Center and LFE stand
/// alone.
pub(crate) fn pairs(channels: usize) -> Vec<(usize, usize)> {
    match channels {
        2 => vec![(0, 1)],
        3 => vec![(0, 2)],
        4 => vec![(0, 1), (2, 3)],
        5..=7 => vec![(0, 2), (3, 4)],
        8 => vec![(0, 2), (3, 4), (5, 6)],
        _ => Vec::new(),
    }
}

/// Collects books as the setup is built.
struct Books {
    specs: Vec<BookSpec>,
    coders: Vec<Coder>,
}

impl Books {
    fn add(&mut self, spec: BookSpec) -> Result<usize, CodecError> {
        self.coders.push(Coder::new(&spec.lengths)?);
        self.specs.push(spec);
        Ok(self.specs.len() - 1)
    }

    fn scalar(&mut self, weights: &[f64]) -> Result<usize, CodecError> {
        self.add(BookSpec {
            dimensions: 1,
            lengths: huffman_lengths(weights, MAX_LENGTH),
            lookup: None,
        })
    }

    /// A lattice book where each position follows a geometric model with
    /// `ratios[position % 2]`; a ratio of 0 allows only 0 there.
    fn lattice(&mut self, lattice: Lattice, ratios: [f64; 2]) -> Result<usize, CodecError> {
        let lengths = lattice.lengths(|values| {
            values
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let ratio = ratios[i % 2];
                    if ratio == 0.0 {
                        f64::from(u8::from(v == 0))
                    } else {
                        geometric(v / lattice.step, ratio)
                    }
                })
                .product()
        });
        self.add(lattice.spec(lengths))
    }
}

const fn lattice(dimensions: usize, side: usize, step: i32) -> Lattice {
    Lattice {
        dimensions,
        side,
        step,
    }
}

/// Classifications of one format, cheapest first: the bounds and models
/// of a single pass, and the escape cascade last.
struct ClassModel {
    limits: [i32; 2],
    lattice: Lattice,
    ratios: [f64; 2],
}

/// Format 2 over a coupled pair: magnitude at even, angle at odd.
const COUPLED: [ClassModel; 9] = [
    ClassModel {
        limits: [1, 0],
        lattice: lattice(4, 3, 1),
        ratios: [0.3, 0.0],
    },
    ClassModel {
        limits: [1, 1],
        lattice: lattice(4, 3, 1),
        ratios: [0.45, 0.3],
    },
    ClassModel {
        limits: [2, 0],
        lattice: lattice(4, 5, 1),
        ratios: [0.5, 0.0],
    },
    ClassModel {
        limits: [2, 2],
        lattice: lattice(4, 5, 1),
        ratios: [0.55, 0.4],
    },
    ClassModel {
        limits: [4, 0],
        lattice: lattice(2, 9, 1),
        ratios: [0.7, 0.0],
    },
    ClassModel {
        limits: [4, 4],
        lattice: lattice(2, 9, 1),
        ratios: [0.7, 0.55],
    },
    ClassModel {
        limits: [8, 8],
        lattice: lattice(2, 17, 1),
        ratios: [0.82, 0.7],
    },
    ClassModel {
        limits: [16, 16],
        lattice: lattice(2, 33, 1),
        ratios: [0.9, 0.82],
    },
    ClassModel {
        limits: [MAX_RESIDUE, MAX_RESIDUE],
        lattice: lattice(2, 33, 1),
        ratios: [0.97, 0.97],
    },
];

/// Format 1 over one channel.
const SINGLE: [ClassModel; 7] = [
    ClassModel {
        limits: [1, 1],
        lattice: lattice(4, 3, 1),
        ratios: [0.2, 0.2],
    },
    ClassModel {
        limits: [1, 1],
        lattice: lattice(4, 3, 1),
        ratios: [0.5, 0.5],
    },
    ClassModel {
        limits: [2, 2],
        lattice: lattice(4, 5, 1),
        ratios: [0.55, 0.55],
    },
    ClassModel {
        limits: [4, 4],
        lattice: lattice(2, 9, 1),
        ratios: [0.7, 0.7],
    },
    ClassModel {
        limits: [8, 8],
        lattice: lattice(2, 17, 1),
        ratios: [0.82, 0.82],
    },
    ClassModel {
        limits: [16, 16],
        lattice: lattice(2, 33, 1),
        ratios: [0.9, 0.9],
    },
    ClassModel {
        limits: [MAX_RESIDUE, MAX_RESIDUE],
        lattice: lattice(2, 33, 1),
        ratios: [0.97, 0.97],
    },
];

/// The escape cascade's coarse passes, before its fine book.
const ESCAPE: [Lattice; 2] = [lattice(1, 33, 1089), lattice(1, 33, 33)];

/// How often each classification is expected, the all-zero one first.
fn class_weight(class: usize) -> f64 {
    [6.0, 4.0, 3.0, 3.0, 2.5, 2.0, 1.5, 1.2, 0.8, 0.4]
        .get(class)
        .copied()
        .unwrap_or(0.2)
}

impl Profile {
    /// The configuration for `channels` at `rate`, coding up to
    /// `lowpass` Hz.
    pub(crate) fn new(channels: usize, rate: u32, lowpass: f64) -> Result<Self, CodecError> {
        if !(1..=crate::header::MAX_CHANNELS).contains(&channels) {
            return Err(CodecError::Unsupported {
                what: "Vorbis encoder channel count",
            });
        }
        if !(8_000..=192_000).contains(&rate) {
            return Err(CodecError::Unsupported {
                what: "Vorbis encoder sample rate",
            });
        }
        let blocksizes = if rate <= 50_000 {
            [256, 2048]
        } else {
            [512, 4096]
        };
        let mut books = Books {
            specs: Vec::new(),
            coders: Vec::new(),
        };

        // Floor books: subclass 0 is the value 0 alone; the others take
        // the values below their entry counts, sparse below the previous.
        let masterbook = {
            let sub = [0.4, 0.4, 0.15, 0.05];
            let weights: Vec<f64> = (0..64)
                .map(|cval: usize| {
                    (0..FLOOR_DIMENSIONS)
                        .map(|k| sub[(cval >> (2 * k)) & 3])
                        .product()
                })
                .collect();
            books.scalar(&weights)?
        };
        let limits = [1, 8, 32, FLOOR_RANGE];
        let mut subbooks = [None; 4];
        for sub in 1..4 {
            let low = limits[sub - 1];
            let weights: Vec<f64> = (0..limits[sub])
                .map(|v| {
                    if v < low {
                        0.0
                    } else {
                        geometric(v - low, [0.0, 0.7, 0.88, 0.97][sub])
                    }
                })
                .collect();
            subbooks[sub] = Some(books.scalar(&weights)?);
        }
        let floor_class = Floor1Class {
            dimensions: FLOOR_DIMENSIONS as u8,
            subclass_bits: 2,
            masterbook: masterbook as u8,
            subclass_books: subbooks.iter().map(|b| b.map(|b| b as u8)).collect(),
        };
        let floor_plan =
            |interior: &[u16], bins: usize, base: usize| -> Result<FloorPlan, CodecError> {
                let scale = (bins / base).max(1) as u16;
                let xs = stream_order(interior, scale, bins as u16);
                let partitions = interior.len() / FLOOR_DIMENSIONS;
                let floor = Floor1::new(
                    vec![0; partitions],
                    vec![floor_class.clone()],
                    FLOOR_MULTIPLIER,
                    bins.trailing_zeros() as u8,
                    xs,
                )
                .ok_or(CodecError::InvalidApi {
                    what: "Vorbis floor points",
                })?;
                Ok(FloorPlan {
                    floor,
                    masterbook,
                    subbooks,
                    limits,
                })
            };
        let floors = [
            floor_plan(&SHORT_POSTS, blocksizes[0] / 2, 128)?,
            floor_plan(&LONG_POSTS, blocksizes[1] / 2, 1024)?,
        ];

        // Residue books, shared by both block sizes.
        let mut class_books = |models: &[ClassModel]| -> Result<Vec<ClassPlan>, CodecError> {
            let mut classes = vec![ClassPlan {
                limits: [0, 0],
                passes: Vec::new(),
            }];
            for model in models {
                let fine = books.lattice(model.lattice, model.ratios)?;
                let mut passes = Vec::new();
                if model.limits[0] == MAX_RESIDUE {
                    for coarse in ESCAPE {
                        passes.push((books.lattice(coarse, [0.25, 0.25])?, coarse));
                    }
                }
                passes.push((fine, model.lattice));
                classes.push(ClassPlan {
                    limits: model.limits,
                    passes,
                });
            }
            Ok(classes)
        };
        let single = class_books(&SINGLE)?;
        let coupled = class_books(&COUPLED)?;
        let per_word = 2;
        let mut classbook = |classes: usize| -> Result<usize, CodecError> {
            let entries = classes.pow(per_word as u32);
            let weights: Vec<f64> = (0..entries)
                .map(|word| {
                    let mut word = word;
                    let mut weight = 1.0;
                    for _ in 0..per_word {
                        weight *= class_weight(word % classes);
                        word /= classes;
                    }
                    weight
                })
                .collect();
            // The classbook's dimension is how many partitions one codeword
            // classifies. The entry index is those classes packed in base
            // `classes`; the book has no vector lookup.
            books.add(BookSpec {
                dimensions: per_word,
                lengths: huffman_lengths(&weights, MAX_LENGTH),
                lookup: None,
            })
        };
        let single_book = classbook(single.len())?;
        let coupled_book = classbook(coupled.len())?;

        let nyquist = f64::from(rate) / 2.0;
        let lowpass = lowpass.clamp(1_000.0, nyquist);
        let bandwidth = [0, 1].map(|size| {
            let bins = blocksizes[size] / 2;
            ((lowpass / nyquist * bins as f64).ceil() as usize).min(bins)
        });
        let residue = |size: usize, kind: u8| -> ResiduePlan {
            let (partition, classes, classbook, width) = if kind == 2 {
                (32, coupled.clone(), coupled_book, 2)
            } else {
                (16, single.clone(), single_book, 1)
            };
            let end = (bandwidth[size] * width).div_ceil(partition) * partition;
            ResiduePlan {
                kind,
                begin: 0,
                end: end.min(blocksizes[size] / 2 * width),
                partition,
                classes,
                classbook,
                per_word,
            }
        };
        let residues = [
            [residue(0, 1), residue(0, 2)],
            [residue(1, 1), residue(1, 2)],
        ];

        let coupling = pairs(channels);
        let mut groups: Vec<Group> = coupling
            .iter()
            .map(|&(a, b)| Group {
                channels: vec![a, b],
                coupled: true,
            })
            .collect();
        let lone: Vec<usize> = (0..channels)
            .filter(|c| !coupling.iter().any(|&(a, b)| a == *c || b == *c))
            .collect();
        if !lone.is_empty() {
            groups.push(Group {
                channels: lone,
                coupled: false,
            });
        }
        let mut mux = vec![0_u8; channels];
        for (index, group) in groups.iter().enumerate() {
            for &channel in &group.channels {
                mux[channel] = index as u8;
            }
        }
        let mapping = |size: usize| Mapping {
            coupling: coupling.iter().map(|&(a, b)| (a as u8, b as u8)).collect(),
            mux: mux.clone(),
            submaps: groups
                .iter()
                .map(|group| (size as u8, (size * 2 + usize::from(group.coupled)) as u8))
                .collect(),
        };
        let setup = SetupSpec {
            books: books.specs,
            floors: vec![
                Floor::One(floors[0].floor.clone()),
                Floor::One(floors[1].floor.clone()),
            ],
            residues: residues
                .iter()
                .flatten()
                .map(|plan| Residue {
                    kind: plan.kind,
                    begin: plan.begin as u32,
                    end: plan.end as u32,
                    partition_size: plan.partition as u32,
                    classifications: plan.classes.len() as u8,
                    classbook: plan.classbook as u8,
                    books: plan
                        .classes
                        .iter()
                        .map(|class| {
                            let mut passes = [None; 8];
                            for (pass, &(book, _)) in class.passes.iter().enumerate() {
                                passes[pass] = Some(book as u8);
                            }
                            passes
                        })
                        .collect(),
                })
                .collect(),
            mappings: vec![mapping(0), mapping(1)],
            modes: vec![
                Mode {
                    long: false,
                    mapping: 0,
                },
                Mode {
                    long: true,
                    mapping: 1,
                },
            ],
        };
        Ok(Self {
            channels,
            rate,
            blocksizes,
            coders: books.coders,
            floors,
            residues,
            groups,
            bandwidth,
            setup,
        })
    }

    /// The identification header, with `nominal` bits per second.
    pub(crate) fn identification(&self, nominal: u32) -> Identification {
        Identification {
            channels: self.channels as u8,
            sample_rate: self.rate,
            bitrate_maximum: 0,
            bitrate_nominal: nominal.min(i32::MAX as u32) as i32,
            bitrate_minimum: 0,
            blocksizes: self.blocksizes,
        }
    }
}

/// Floor X values in stream order: the two ends, then each interval's
/// middle point breadth first, so every point is predicted from close
/// neighbors that are already coded.
fn stream_order(interior: &[u16], scale: u16, end: u16) -> Vec<u16> {
    let mut xs = vec![0, end];
    let mut queue = std::collections::VecDeque::from([(0_usize, interior.len())]);
    while let Some((low, high)) = queue.pop_front() {
        if low >= high {
            continue;
        }
        let middle = (low + high) / 2;
        xs.push(interior[middle] * scale);
        queue.push_back((low, middle));
        queue.push_back((middle + 1, high));
    }
    xs
}

#[cfg(test)]
mod tests {
    use super::{Profile, pairs, stream_order};
    use crate::setup::{Setup, SetupSpec};

    #[test]
    fn the_setup_parses_back_for_every_channel_count_and_rate() {
        for channels in 1..=8 {
            for rate in [8_000, 22_050, 44_100, 48_000, 96_000] {
                let profile = Profile::new(channels, rate, 18_000.0).expect("profile");
                let packet = profile.setup.write(channels);
                let parsed =
                    SetupSpec::parse(&packet, channels, profile.blocksizes).expect("parse");
                assert_eq!(parsed, profile.setup, "{channels} channels at {rate}");
                Setup::parse(&packet, channels, profile.blocksizes).expect("build");
                let coupled = pairs(channels).len();
                assert_eq!(profile.groups.iter().filter(|g| g.coupled).count(), coupled);
                let covered: usize = profile.groups.iter().map(|g| g.channels.len()).sum();
                assert_eq!(covered, channels);
            }
        }
    }

    #[test]
    fn floor_points_come_coarse_to_fine() {
        let xs = stream_order(&[2, 4, 8, 16, 32, 64, 128], 1, 256);
        assert_eq!(xs, vec![0, 256, 16, 4, 64, 2, 8, 32, 128]);
    }

    #[test]
    fn the_residue_stops_at_the_lowpass() {
        let profile = Profile::new(2, 44_100, 11_025.0).expect("profile");
        assert_eq!(profile.bandwidth, [64, 512]);
        assert_eq!(profile.residues[1][1].end, 1024);
        assert_eq!(profile.residues[1][0].end, 512);
        assert!(Profile::new(9, 44_100, 18_000.0).is_err());
        assert!(Profile::new(2, 4_000, 3_000.0).is_err());
    }
}
