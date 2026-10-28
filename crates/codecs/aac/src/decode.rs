//! AAC-LC raw access-unit decoder.
//!
//! A failed frame leaves overlap, the window memory, and the sample clock as
//! they were. Spectral tools run only after the whole frame has parsed.
//! Prediction in LC, coupling, gain control, and spectral band replication are
//! errors rather than something the decoder skips.

use cutlass_decode::{AudioOutput, CodecError, CodecId, CodecParams, Decoder, Packet, Pcm};

use crate::bits::BitReader;
use crate::config::{AudioConfig, ElementKind};
use crate::filterbank::{FULL_SCALE, Overlap, Sequence, Windows, synthesize};
use crate::huffman::{
    pair_digits, quad_digits, read_escape, read_scale_delta, read_spectral as read_huff,
    signed_pair,
};
use crate::quant::{dequant, pow2_quarter};
use crate::tns::{self, TnsWindow};

const END: u32 = 7;

pub struct AacDecoder {
    config: AudioConfig,
    windows: Windows,
    overlaps: Vec<Overlap>,
    frame_index: u64,
    live: bool,
}

impl AacDecoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        let config = AudioConfig::from_params(params)?;
        let windows = Windows::new(config.frame_length)?;
        let overlaps = vec![Overlap::new(config.frame_length); config.layout.len()];
        Ok(Self {
            config,
            windows,
            overlaps,
            frame_index: 0,
            live: false,
        })
    }

    pub fn config(&self) -> &AudioConfig {
        &self.config
    }

    pub fn push_packet(&mut self, packet: &Packet) -> Result<Vec<f32>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        if packet.data.len() > 8_192 {
            return Err(CodecError::Limit {
                what: "access unit bytes",
            });
        }
        for side in &packet.side {
            match side {
                cutlass_decode::SideData::Config(bytes)
                    if bytes.as_slice() != self.config.asc.as_slice() =>
                {
                    return Err(CodecError::Unsupported {
                        what: "configuration change",
                    });
                }
                cutlass_decode::SideData::Protection { crc, header } => {
                    let found = protection_crc(header.as_deref(), &packet.data, &self.config)?;
                    if found != *crc {
                        return Err(CodecError::Malformed {
                            at_bit: 0,
                            what: "crc",
                        });
                    }
                }
                _ => {}
            }
        }
        let mut scratch = self.overlaps.clone();
        let pcm = decode_unit(&self.config, &self.windows, &mut scratch, &packet.data)?;
        self.overlaps = scratch;
        self.frame_index = self.frame_index.saturating_add(1);
        self.live = true;
        Ok(pcm)
    }

    pub fn drain_pcm(&mut self) -> Result<Option<Vec<f32>>, CodecError> {
        if !self.live || self.overlaps.iter().all(Overlap::is_silent) {
            self.live = false;
            return Ok(None);
        }
        let mut scratch = self.overlaps.clone();
        let mut channels = Vec::with_capacity(scratch.len());
        for overlap in &mut scratch {
            let seq = overlap.flush_sequence();
            let kbd = overlap.prev_shape_kbd();
            let spec = vec![0.0; self.config.frame_length];
            channels.push(synthesize(&self.windows, overlap, &spec, seq, kbd)?);
        }
        self.overlaps = scratch;
        self.frame_index = self.frame_index.saturating_add(1);
        self.live = false;
        Ok(Some(interleave(&channels)))
    }

    pub fn reset_state(&mut self) {
        let frame = self.config.frame_length;
        self.overlaps = vec![Overlap::new(frame); self.config.layout.len()];
        self.frame_index = 0;
        self.live = false;
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn codec_id(&self) -> CodecId {
        CodecId::Aac
    }
}

fn decode_unit(
    config: &AudioConfig,
    windows: &Windows,
    overlaps: &mut [Overlap],
    data: &[u8],
) -> Result<Vec<f32>, CodecError> {
    let mut bits = BitReader::new(data);
    let mut channels = Vec::new();
    let mut produced = 0usize;
    let mut seen = 0usize;
    while bits.bits_left() >= 3 {
        seen += 1;
        if seen > 32 {
            return Err(CodecError::Limit { what: "elements" });
        }
        let id = bits.read(3)?;
        if id == END {
            break;
        }
        match id {
            0 => {
                expect(config, &mut produced, ElementKind::Sce, &bits)?;
                channels.push(decode_single(&mut bits, config, false)?);
            }
            1 => {
                expect(config, &mut produced, ElementKind::Cpe, &bits)?;
                let pair = decode_pair(&mut bits, config)?;
                channels.extend(pair);
            }
            2 => {
                return Err(CodecError::Unsupported { what: "coupling" });
            }
            3 => {
                expect(config, &mut produced, ElementKind::Lfe, &bits)?;
                channels.push(decode_single(&mut bits, config, true)?);
            }
            4 => skip_data(&mut bits)?,
            5 => {
                return Err(CodecError::Unsupported {
                    what: "program config",
                });
            }
            6 => skip_fill(&mut bits)?,
            _ => {
                return Err(malformed(&bits, "element"));
            }
        }
    }
    if produced != config.elements.len() {
        return Err(malformed(&bits, "channel"));
    }
    if channels.len() != overlaps.len() {
        return Err(malformed(&bits, "channel"));
    }
    let mut pcm = Vec::with_capacity(channels.len());
    for (channel, overlap) in channels.iter().zip(overlaps.iter_mut()) {
        let mut spec = channel.spec.clone();
        for bin in &mut spec {
            *bin /= FULL_SCALE;
        }
        pcm.push(synthesize(
            windows,
            overlap,
            &spec,
            channel.ics.seq,
            channel.ics.kbd,
        )?);
    }
    Ok(interleave(&pcm))
}

fn expect(
    config: &AudioConfig,
    produced: &mut usize,
    kind: ElementKind,
    bits: &BitReader<'_>,
) -> Result<(), CodecError> {
    match config.elements.get(*produced) {
        Some(expected) if *expected == kind => {
            *produced += 1;
            Ok(())
        }
        _ => Err(malformed(bits, "element")),
    }
}

#[derive(Clone)]
struct Ics {
    seq: Sequence,
    kbd: bool,
    max_sfb: usize,
    groups: Vec<usize>,
    short: bool,
}

struct Channel {
    spec: Vec<f32>,
    ics: Ics,
    books: Vec<u8>,
    scales: Vec<i32>,
    tns: Vec<TnsWindow>,
}

fn decode_single(
    bits: &mut BitReader<'_>,
    config: &AudioConfig,
    _lfe: bool,
) -> Result<Channel, CodecError> {
    let _tag = bits.read(4)?;
    let mut channel = decode_ics(bits, config, None)?;
    if channel.books.iter().any(|book| *book == 14 || *book == 15) {
        return Err(malformed(bits, "intensity"));
    }
    apply_noise(config, &mut channel, &mut 0x1F2E_3D4C)?;
    apply_tns_channel(config, &mut channel)?;
    Ok(channel)
}

fn decode_pair(bits: &mut BitReader<'_>, config: &AudioConfig) -> Result<Vec<Channel>, CodecError> {
    let _tag = bits.read(4)?;
    let common = bits.read(1)? == 1;
    let (shared, ms) = if common {
        let ics = read_ics_info(bits, config)?;
        let ms = read_ms(bits, &ics)?;
        (Some(ics), ms)
    } else {
        (None, Vec::new())
    };
    let mut left = decode_ics(bits, config, shared.clone())?;
    let mut right = decode_ics(bits, config, shared)?;
    if common {
        apply_stereo(config, &mut left, &mut right, &ms)?;
    } else {
        apply_noise(config, &mut left, &mut 0x1F2E_3D4C)?;
        apply_noise(config, &mut right, &mut 0x1F2E_3D4C)?;
    }
    apply_tns_channel(config, &mut left)?;
    apply_tns_channel(config, &mut right)?;
    Ok(vec![left, right])
}

fn decode_ics(
    bits: &mut BitReader<'_>,
    config: &AudioConfig,
    shared: Option<Ics>,
) -> Result<Channel, CodecError> {
    let global_gain = bits.read(8)?;
    let ics = if let Some(shared) = shared {
        shared
    } else {
        read_ics_info(bits, config)?
    };
    let bands = config.bands(ics.short)?;
    let num_swb = bands.len().saturating_sub(1);
    if ics.max_sfb > num_swb {
        return Err(malformed(bits, "scale factor band"));
    }
    let books = read_sections(bits, &ics)?;
    let scales = read_scales(bits, global_gain, &ics, &books)?;
    let mut quant = vec![0i32; config.frame_length];
    if bits.read(1)? == 1 {
        if ics.short {
            return Err(malformed(bits, "pulse"));
        }
        read_pulses(bits, &mut quant, bands, config.frame_length)?;
    }
    let tns_present = bits.read(1)? == 1;
    let tns = if tns_present {
        let windows = if ics.short { 8 } else { 1 };
        tns::read_tns(bits, windows, ics.short)?
    } else {
        Vec::new()
    };
    if bits.read(1)? == 1 {
        return Err(CodecError::Unsupported {
            what: "gain control",
        });
    }
    decode_coefficients(bits, &ics, &books, bands, &mut quant)?;
    let spec = dequantize(&quant, &ics, &books, &scales, bands);
    Ok(Channel {
        spec,
        ics,
        books,
        scales,
        tns,
    })
}

fn read_ics_info(bits: &mut BitReader<'_>, config: &AudioConfig) -> Result<Ics, CodecError> {
    if bits.read(1)? != 0 {
        return Err(malformed(bits, "ics reserved"));
    }
    let seq = Sequence::from_bits(bits.read(2)?)?;
    let kbd = bits.read(1)? == 1;
    let short = seq == Sequence::EightShort;
    let max_sfb = bits.read(if short { 4 } else { 6 })? as usize;
    let groups = if short {
        group_lengths(bits.read(7)?)
    } else {
        if bits.read(1)? == 1 {
            return Err(malformed(bits, "prediction"));
        }
        vec![1]
    };
    if short && config.frame_length != 1024 && config.frame_length != 960 {
        return Err(CodecError::Unsupported {
            what: "frame length",
        });
    }
    Ok(Ics {
        seq,
        kbd,
        max_sfb,
        groups,
        short,
    })
}

fn group_lengths(grouping: u32) -> Vec<usize> {
    let mut lengths = Vec::new();
    let mut len = 1usize;
    for shift in (0..7).rev() {
        if (grouping >> shift) & 1 == 1 {
            len += 1;
        } else {
            lengths.push(len);
            len = 1;
        }
    }
    lengths.push(len);
    lengths
}

fn read_ms(bits: &mut BitReader<'_>, ics: &Ics) -> Result<Vec<bool>, CodecError> {
    let count = ics.groups.len() * ics.max_sfb;
    match bits.read(2)? {
        0 => Ok(vec![false; count]),
        2 => Ok(vec![true; count]),
        1 => {
            let mut mask = Vec::with_capacity(count);
            for _ in 0..count {
                mask.push(bits.read(1)? == 1);
            }
            Ok(mask)
        }
        _ => Err(malformed(bits, "ms mask")),
    }
}

fn read_sections(bits: &mut BitReader<'_>, ics: &Ics) -> Result<Vec<u8>, CodecError> {
    if ics.max_sfb == 0 {
        return Ok(Vec::new());
    }
    let mut books = vec![0u8; ics.groups.len() * ics.max_sfb];
    let sect_bits = if ics.short { 3 } else { 5 };
    let escape = (1u32 << sect_bits) - 1;
    for slot in books.chunks_mut(ics.max_sfb) {
        let mut sfb = 0usize;
        while sfb < ics.max_sfb {
            let book = u8::try_from(bits.read(4)?).unwrap_or(255);
            if book == 12 || book > 15 {
                return Err(malformed(bits, "codebook"));
            }
            let mut len = 0usize;
            loop {
                let incr = bits.read(sect_bits)? as usize;
                len += incr;
                if len > ics.max_sfb {
                    return Err(malformed(bits, "section"));
                }
                if u32::try_from(incr).unwrap_or(escape) != escape {
                    break;
                }
                if incr == 0 {
                    return Err(malformed(bits, "section"));
                }
            }
            if len == 0 || sfb + len > ics.max_sfb {
                return Err(malformed(bits, "section"));
            }
            for book_slot in slot.iter_mut().skip(sfb).take(len) {
                *book_slot = book;
            }
            sfb += len;
        }
    }
    Ok(books)
}

fn read_scales(
    bits: &mut BitReader<'_>,
    global_gain: u32,
    ics: &Ics,
    books: &[u8],
) -> Result<Vec<i32>, CodecError> {
    let mut scales = vec![0i32; books.len()];
    let mut sf = i32::try_from(global_gain).unwrap_or(0);
    let mut noise = sf - 90;
    let mut noise_pcm = true;
    let mut intensity = 0i32;
    for (index, &book) in books.iter().enumerate() {
        match book {
            0 => {}
            13 => {
                if noise_pcm {
                    noise = i32::try_from(bits.read(9)?).unwrap_or(0) - 256;
                    noise_pcm = false;
                } else {
                    noise += read_scale_delta(bits)?;
                }
                if !(-100..=155).contains(&noise) {
                    return Err(malformed(bits, "scale factor"));
                }
                scales[index] = noise;
            }
            14 | 15 => {
                intensity += read_scale_delta(bits)?;
                if !(-155..=100).contains(&intensity) {
                    return Err(malformed(bits, "scale factor"));
                }
                scales[index] = intensity;
            }
            _ => {
                sf += read_scale_delta(bits)?;
                if !(0..=255).contains(&sf) {
                    return Err(malformed(bits, "scale factor"));
                }
                scales[index] = sf;
            }
        }
    }
    let _ = ics;
    Ok(scales)
}

fn read_pulses(
    bits: &mut BitReader<'_>,
    quant: &mut [i32],
    bands: &[u16],
    frame: usize,
) -> Result<(), CodecError> {
    let count = bits.read(2)? as usize + 1;
    let start = bits.read(6)? as usize;
    let Some(&origin) = bands.get(start) else {
        return Err(malformed(bits, "pulse"));
    };
    let mut pos = i32::from(origin);
    for _ in 0..count {
        pos += i32::try_from(bits.read(5)?).unwrap_or(0);
        let amp = i32::try_from(bits.read(4)?).unwrap_or(0);
        let Some(bin) = usize::try_from(pos).ok().filter(|bin| *bin < frame) else {
            return Err(malformed(bits, "pulse"));
        };
        if quant[bin] > 0 {
            quant[bin] += amp;
        } else {
            quant[bin] -= amp;
        }
    }
    Ok(())
}

fn decode_coefficients(
    bits: &mut BitReader<'_>,
    ics: &Ics,
    books: &[u8],
    bands: &[u16],
    quant: &mut [i32],
) -> Result<(), CodecError> {
    let bins = if ics.short {
        quant.len() / 8
    } else {
        quant.len()
    };
    let mut windows_at = 0usize;
    for (group, group_len) in ics.groups.iter().copied().enumerate() {
        for sfb in 0..ics.max_sfb {
            let book = books[group * ics.max_sfb + sfb];
            let start = usize::from(*bands.get(sfb).ok_or(malformed(bits, "scale factor band"))?);
            let end = usize::from(
                *bands
                    .get(sfb + 1)
                    .ok_or(malformed(bits, "scale factor band"))?,
            );
            if end < start || end > bins {
                return Err(malformed(bits, "scale factor band"));
            }
            for window in 0..group_len {
                let offset = (windows_at + window) * bins;
                let dest = quant
                    .get_mut(offset + start..offset + end)
                    .ok_or(malformed(bits, "spectral"))?;
                read_band(bits, book, dest)?;
            }
        }
        windows_at += group_len;
    }
    Ok(())
}

fn read_band(bits: &mut BitReader<'_>, book: u8, dest: &mut [i32]) -> Result<(), CodecError> {
    match book {
        0 | 13 | 14 | 15 => Ok(()),
        1 | 2 => read_quads(bits, book, dest, true),
        3 | 4 => read_quads(bits, book, dest, false),
        5 | 6 => read_pairs(bits, book, dest, 9, false),
        7 | 8 => read_pairs(bits, book, dest, 8, false),
        9 | 10 => read_pairs(bits, book, dest, 13, false),
        11 => read_pairs(bits, book, dest, 17, true),
        _ => Err(malformed(bits, "codebook")),
    }
}

fn read_quads(
    bits: &mut BitReader<'_>,
    book: u8,
    dest: &mut [i32],
    signed: bool,
) -> Result<(), CodecError> {
    if !dest.len().is_multiple_of(4) {
        return Err(malformed(bits, "spectral"));
    }
    for chunk in dest.chunks_mut(4) {
        let index = read_huff(bits, usize::from(book - 1))?;
        let digits = quad_digits(index);
        for (slot, digit) in chunk.iter_mut().zip(digits) {
            let mut value = if signed { digit - 1 } else { digit };
            if !signed && value != 0 && bits.read(1)? == 1 {
                value = -value;
            }
            *slot = value;
        }
    }
    Ok(())
}

fn read_pairs(
    bits: &mut BitReader<'_>,
    book: u8,
    dest: &mut [i32],
    modulus: i32,
    escape: bool,
) -> Result<(), CodecError> {
    if !dest.len().is_multiple_of(2) {
        return Err(malformed(bits, "spectral"));
    }
    for chunk in dest.chunks_mut(2) {
        let index = read_huff(bits, usize::from(book - 1))?;
        let digits = if book == 5 || book == 6 {
            signed_pair(index)
        } else {
            pair_digits(index, modulus)
        };
        let mut values = digits;
        let mut signs = [1i32, 1];
        if book != 5 && book != 6 {
            for (value, sign) in values.iter().zip(signs.iter_mut()) {
                if *value != 0 && bits.read(1)? == 1 {
                    *sign = -1;
                }
            }
            if escape {
                for value in &mut values {
                    if *value == 16 {
                        *value = read_escape(bits)?;
                    }
                }
            }
        }
        chunk[0] = values[0] * signs[0];
        chunk[1] = values[1] * signs[1];
    }
    Ok(())
}

fn dequantize(quant: &[i32], ics: &Ics, books: &[u8], scales: &[i32], bands: &[u16]) -> Vec<f32> {
    let mut spec = vec![0.0; quant.len()];
    let bins = if ics.short {
        quant.len() / 8
    } else {
        quant.len()
    };
    let mut windows_at = 0usize;
    for (group, group_len) in ics.groups.iter().copied().enumerate() {
        for sfb in 0..ics.max_sfb {
            let index = group * ics.max_sfb + sfb;
            let book = books[index];
            if !(1..=11).contains(&book) {
                continue;
            }
            let start = usize::from(bands[sfb]);
            let end = usize::from(bands[sfb + 1]);
            let scale = scales[index];
            for window in 0..group_len {
                let offset = (windows_at + window) * bins;
                for bin in start..end {
                    spec[offset + bin] = dequant(quant[offset + bin], scale);
                }
            }
        }
        windows_at += group_len;
    }
    spec
}

fn apply_tns_channel(config: &AudioConfig, channel: &mut Channel) -> Result<(), CodecError> {
    if channel.tns.is_empty() {
        return Ok(());
    }
    let bands = config.bands(channel.ics.short)?;
    let window_bins = if channel.ics.short {
        config.frame_length / 8
    } else {
        config.frame_length
    };
    tns::apply(
        &mut channel.spec,
        &channel.tns,
        bands,
        config.tns_max_bands(channel.ics.short),
        channel.ics.max_sfb,
        window_bins,
    );
    Ok(())
}

fn apply_noise(
    config: &AudioConfig,
    channel: &mut Channel,
    state: &mut u32,
) -> Result<(), CodecError> {
    let bands = config.bands(channel.ics.short)?;
    let bins = window_bins(channel);
    let groups = channel.ics.groups.clone();
    let max_sfb = channel.ics.max_sfb;
    let mut windows_at = 0usize;
    for (group, group_len) in groups.into_iter().enumerate() {
        for sfb in 0..max_sfb {
            let index = group * max_sfb + sfb;
            let book = channel.books[index];
            if book != 13 {
                continue;
            }
            let scale = pow2_quarter(channel.scales[index]);
            let (start, end) = band_bins(bands, sfb);
            for window in 0..group_len {
                let offset = (windows_at + window) * bins;
                fill_noise(
                    state,
                    scale,
                    &mut channel.spec[offset + start..offset + end],
                );
            }
        }
        windows_at += group_len;
    }
    Ok(())
}

fn apply_stereo(
    config: &AudioConfig,
    left: &mut Channel,
    right: &mut Channel,
    ms: &[bool],
) -> Result<(), CodecError> {
    if left.ics.max_sfb != right.ics.max_sfb || left.ics.groups != right.ics.groups {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "common window",
        });
    }
    let bands = config.bands(left.ics.short)?;
    let bins = window_bins(left);
    let groups = left.ics.groups.clone();
    let max_sfb = left.ics.max_sfb;
    let mut left_state = 0x1F2E_3D4Cu32;
    let mut right_state = left_state;
    let mut windows_at = 0usize;
    for (group, group_len) in groups.into_iter().enumerate() {
        for sfb in 0..max_sfb {
            let index = group * max_sfb + sfb;
            let left_book = left.books[index];
            let right_book = right.books[index];
            let ms_used = ms.get(index).copied().unwrap_or(false);
            let (start, end) = band_bins(bands, sfb);
            for window in 0..group_len {
                let offset = (windows_at + window) * bins;
                let left_band = offset + start..offset + end;
                let right_band = left_band.clone();
                mix_band(BandMix {
                    left,
                    right,
                    left_band,
                    right_band,
                    left_book,
                    right_book,
                    index,
                    ms_used,
                    left_state: &mut left_state,
                    right_state: &mut right_state,
                })?;
            }
        }
        windows_at += group_len;
    }
    Ok(())
}

struct BandMix<'a> {
    left: &'a mut Channel,
    right: &'a mut Channel,
    left_band: std::ops::Range<usize>,
    right_band: std::ops::Range<usize>,
    left_book: u8,
    right_book: u8,
    index: usize,
    ms_used: bool,
    left_state: &'a mut u32,
    right_state: &'a mut u32,
}

fn mix_band(mix: BandMix<'_>) -> Result<(), CodecError> {
    let BandMix {
        left,
        right,
        left_band,
        right_band,
        left_book,
        right_book,
        index,
        ms_used,
        left_state,
        right_state,
    } = mix;
    if left_book == 14 || left_book == 15 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "intensity",
        });
    }
    if right_book == 14 || right_book == 15 {
        let mut scale = pow2_quarter(100 - right.scales[index]);
        if right_book == 14 {
            scale = -scale;
        }
        if ms_used {
            scale = -scale;
        }
        let source = left.spec[left_band].to_vec();
        for (dest, sample) in right.spec[right_band].iter_mut().zip(source) {
            *dest = sample * scale;
        }
        return Ok(());
    }
    if left_book == 13 && right_book == 13 && ms_used {
        let left_scale = pow2_quarter(left.scales[index]);
        let right_scale = pow2_quarter(right.scales[index]);
        fill_noise(left_state, left_scale, &mut left.spec[left_band.clone()]);
        right.spec[right_band.clone()].copy_from_slice(&left.spec[left_band]);
        if left_scale.abs() > 1.0e-12 {
            let ratio = right_scale / left_scale;
            for sample in &mut right.spec[right_band] {
                *sample *= ratio;
            }
        }
        *right_state = *left_state;
        return Ok(());
    }
    if left_book == 13 {
        fill_noise(
            left_state,
            pow2_quarter(left.scales[index]),
            &mut left.spec[left_band.clone()],
        );
    }
    if right_book == 13 {
        fill_noise(
            right_state,
            pow2_quarter(right.scales[index]),
            &mut right.spec[right_band.clone()],
        );
    }
    if ms_used {
        let mid = left.spec[left_band.clone()].to_vec();
        let side = right.spec[right_band.clone()].to_vec();
        for (dest, (left_sample, right_sample)) in
            left.spec[left_band].iter_mut().zip(mid.iter().zip(&side))
        {
            *dest = left_sample + right_sample;
        }
        for (dest, (left_sample, right_sample)) in
            right.spec[right_band].iter_mut().zip(mid.iter().zip(side))
        {
            *dest = left_sample - right_sample;
        }
    }
    Ok(())
}

fn window_bins(channel: &Channel) -> usize {
    if channel.ics.short {
        channel.spec.len() / 8
    } else {
        channel.spec.len()
    }
}

fn band_bins(bands: &[u16], sfb: usize) -> (usize, usize) {
    (usize::from(bands[sfb]), usize::from(bands[sfb + 1]))
}

fn fill_noise(state: &mut u32, scale: f32, dest: &mut [f32]) {
    let mut energy = 0.0f64;
    for sample in dest.iter_mut() {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = f32::from(((*state >> 16) as u16) as i16);
        *sample = unit;
        energy += f64::from(unit) * f64::from(unit);
    }
    let gain = if energy <= 0.0 {
        0.0
    } else {
        f64::from(scale) / energy.sqrt()
    };
    for sample in dest {
        *sample = (f64::from(*sample) * gain) as f32;
    }
}

fn skip_data(bits: &mut BitReader<'_>) -> Result<(), CodecError> {
    let _tag = bits.read(4)?;
    let align = bits.read(1)? == 1;
    let mut count = bits.read(8)?;
    if count == 255 {
        count += bits.read(8)?;
    }
    if count > 512 {
        return Err(CodecError::Limit {
            what: "data stream",
        });
    }
    if align {
        bits.byte_align()?;
    }
    for _ in 0..count {
        let _byte = bits.read(8)?;
    }
    Ok(())
}

fn skip_fill(bits: &mut BitReader<'_>) -> Result<(), CodecError> {
    let mut count = bits.read(4)?;
    if count == 15 {
        let escape = bits.read(8)?;
        count = count + escape - 1;
    }
    if count > 270 {
        return Err(CodecError::Limit { what: "fill" });
    }
    let payload_bits = usize::try_from(count).unwrap_or(0) * 8;
    if bits.bits_left() < payload_bits {
        return Err(CodecError::Truncated {
            at_bit: bits.position() as u64,
            what: "fill",
        });
    }
    if payload_bits >= 4 {
        let extension = bits.read(4)?;
        if extension == 13 || extension == 14 {
            return Err(CodecError::Unsupported { what: "sbr" });
        }
        let rest = payload_bits - 4;
        if rest > 0 {
            skip_bits(bits, rest)?;
        }
    }
    Ok(())
}

/// CRC-16, polynomial x^16 + x^15 + x^2 + 1, register starting at `0xFFFF`.
///
/// ISO/IEC 13818-7 points this at ISO/IEC 11172-3. A single-block ADTS CRC
/// covers the 56 header bits and then, per element, the first 192 bits after
/// `id_syn_ele` (128 more from the second channel stream of a pair). A
/// multi-block raw CRC omits the header. Short elements are zero-padded to
/// that length. Data-stream elements are covered in full.
pub fn protection_crc(
    header: Option<&[u8]>,
    raw: &[u8],
    config: &AudioConfig,
) -> Result<u16, CodecError> {
    let mut crc = Crc16::new();
    if let Some(header) = header {
        if header.len() != 7 {
            return Err(CodecError::InvalidApi {
                what: "adts header",
            });
        }
        for byte in header {
            for shift in (0..8).rev() {
                crc.bit((byte >> shift) & 1);
            }
        }
    }
    let mut bits = BitReader::new(raw);
    let mut seen = 0usize;
    while bits.bits_left() >= 3 {
        seen += 1;
        if seen > 32 {
            return Err(CodecError::Limit { what: "elements" });
        }
        let id = bits.read(3)?;
        if id == END {
            break;
        }
        let after = bits.position();
        match id {
            0 | 3 => {
                let _ = decode_single(&mut bits, config, id == 3)?;
                crc.cover(raw, after, bits.position(), 192);
            }
            1 => {
                let second = consume_pair(&mut bits, config)?;
                crc.cover(raw, after, bits.position(), 192);
                crc.cover(raw, second, bits.position(), 128);
            }
            4 => {
                skip_data(&mut bits)?;
                crc.cover(raw, after, bits.position(), bits.position() - after);
            }
            2 => return Err(CodecError::Unsupported { what: "coupling" }),
            5 => {
                return Err(CodecError::Unsupported {
                    what: "program config",
                });
            }
            6 => skip_fill(&mut bits)?,
            _ => return Err(malformed(&bits, "element")),
        }
    }
    Ok(crc.finish())
}

fn consume_pair(bits: &mut BitReader<'_>, config: &AudioConfig) -> Result<usize, CodecError> {
    let _tag = bits.read(4)?;
    let common = bits.read(1)? == 1;
    let shared = if common {
        let ics = read_ics_info(bits, config)?;
        let _ms = read_ms(bits, &ics)?;
        Some(ics)
    } else {
        None
    };
    let _left = decode_ics(bits, config, shared.clone())?;
    let second = bits.position();
    let _right = decode_ics(bits, config, shared)?;
    Ok(second)
}

struct Crc16 {
    reg: u16,
}

impl Crc16 {
    fn new() -> Self {
        Self { reg: 0xFFFF }
    }

    fn bit(&mut self, bit: u8) {
        let out = (self.reg >> 15) & 1;
        self.reg <<= 1;
        if out ^ u16::from(bit & 1) != 0 {
            self.reg ^= 0x8005;
        }
    }

    fn cover(&mut self, data: &[u8], start: usize, end: usize, width: usize) {
        let available = end.saturating_sub(start);
        let take = available.min(width);
        for index in 0..take {
            let at = start + index;
            let byte = data.get(at / 8).copied().unwrap_or(0);
            let shift = 7 - (at % 8);
            self.bit((byte >> shift) & 1);
        }
        for _ in take..width {
            self.bit(0);
        }
    }

    fn finish(self) -> u16 {
        self.reg
    }
}

fn skip_bits(bits: &mut BitReader<'_>, mut count: usize) -> Result<(), CodecError> {
    while count > 0 {
        let take = u32::try_from(count.min(24)).unwrap_or(24);
        let _ = bits.read(take)?;
        count -= take as usize;
    }
    Ok(())
}

fn interleave(channels: &[Vec<f32>]) -> Vec<f32> {
    let Some(first) = channels.first() else {
        return Vec::new();
    };
    let mut out = vec![0.0; first.len() * channels.len()];
    for (channel, pcm) in channels.iter().enumerate() {
        for (index, sample) in pcm.iter().enumerate() {
            out[index * channels.len() + channel] = *sample;
        }
    }
    out
}

fn malformed(bits: &BitReader<'_>, what: &'static str) -> CodecError {
    CodecError::Malformed {
        at_bit: bits.position() as u64,
        what,
    }
}

impl AacDecoder {
    fn output(&self, samples: Vec<f32>, index: u64) -> Result<AudioOutput, CodecError> {
        let pcm = Pcm::from_f32(self.config.sample_rate, self.config.layout.clone(), samples)
            .map_err(|_| CodecError::Limit { what: "pcm" })?;
        let frame = u64::try_from(self.config.frame_length).unwrap_or(0);
        let pts = i64::try_from(index.saturating_mul(frame)).unwrap_or(i64::MAX);
        Ok(AudioOutput {
            pcm,
            pts,
            timescale: self.config.sample_rate,
            skip_leading: 0,
            skip_trailing: 0,
            concealed: false,
        })
    }
}

impl Decoder for AacDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Aac
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<AudioOutput>, CodecError> {
        let samples = self.push_packet(packet)?;
        let index = self.frame_index.saturating_sub(1);
        Ok(vec![self.output(samples, index)?])
    }

    fn drain(&mut self) -> Result<Vec<AudioOutput>, CodecError> {
        let Some(samples) = self.drain_pcm()? else {
            return Ok(Vec::new());
        };
        let index = self.frame_index.saturating_sub(1);
        Ok(vec![self.output(samples, index)?])
    }

    fn reset(&mut self) {
        self.reset_state();
    }
}

#[cfg(test)]
mod tests {
    use super::group_lengths;
    use crate::bits::{BitSink, BitWriter};
    use crate::decode::AacDecoder;
    use cutlass_decode::{ChannelLayout, CodecError, CodecId, CodecParams, Packet, SideData};

    #[test]
    fn grouping_bits_follow_transmission_order() {
        assert_eq!(group_lengths(0), vec![1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(group_lengths(0b111_1111), vec![8]);
        assert_eq!(group_lengths(0b100_0000), vec![2, 1, 1, 1, 1, 1, 1]);
    }

    fn silent_frame() -> Vec<u8> {
        let mut writer = BitWriter::with_cap(16);
        writer.write_bits(0, 3).unwrap();
        writer.write_bits(0, 4).unwrap();
        writer.write_bits(0, 8).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 2).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 6).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(7, 3).unwrap();
        writer.finish()
    }

    #[test]
    fn a_silent_frame_decodes_to_one_frame_of_zeros() {
        let params = CodecParams::audio(CodecId::Aac, 48_000, ChannelLayout::mono());
        let mut decoder = AacDecoder::open(&params).unwrap();
        let packet = Packet::audio(silent_frame(), 0, 1024, 48_000);
        let pcm = decoder.push_packet(&packet).unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| sample.abs() < 1.0e-6));
        assert!(decoder.drain_pcm().unwrap().is_none());
    }

    #[test]
    fn a_wrong_crc_does_not_advance_the_decoder() {
        let params = CodecParams::audio(CodecId::Aac, 48_000, ChannelLayout::mono());
        let mut decoder = AacDecoder::open(&params).unwrap();
        let raw = silent_frame();
        let header = [0xFFu8, 0xF0, 0x4C, 0x40, 0x00, 0x1F, 0xFC];
        let crc = super::protection_crc(Some(&header), &raw, decoder.config()).unwrap();
        let mut packet = Packet::audio(raw, 0, 1024, 48_000);
        packet.side.push(SideData::Protection {
            crc: crc ^ 1,
            header: Some(header.to_vec()),
        });
        let err = decoder.push_packet(&packet).unwrap_err();
        assert_eq!(
            err,
            CodecError::Malformed {
                at_bit: 0,
                what: "crc"
            }
        );
        assert_eq!(decoder.frame_index(), 0);
        packet.side.clear();
        packet.side.push(SideData::Protection {
            crc,
            header: Some(header.to_vec()),
        });
        assert_eq!(decoder.push_packet(&packet).unwrap().len(), 1024);
    }

    #[test]
    fn a_truncated_frame_keeps_the_decoder() {
        let params = CodecParams::audio(CodecId::Aac, 48_000, ChannelLayout::mono());
        let mut decoder = AacDecoder::open(&params).unwrap();
        let packet = Packet::audio(vec![0x00, 0x00], 0, 1024, 48_000);
        assert!(decoder.push_packet(&packet).is_err());
        assert_eq!(decoder.frame_index(), 0);
    }
}
