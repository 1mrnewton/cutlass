//! Visual object sequence, video object layer, group of VOP, VOP, and
//! short-header picture headers (ISO/IEC 14496-2 Sections 6.2 and 6.3).

use cutlass_decode::CodecError;

use crate::bits::BitReader;
use crate::tables::{
    DEFAULT_INTER_MATRIX, DEFAULT_INTRA_MATRIX, INTRA_DC_THRESHOLD, ZIGZAG, tables,
};

/// `visual_object_sequence_start_code`.
pub(crate) const VOS_START: u8 = 0xB0;
/// `visual_object_sequence_end_code`.
pub(crate) const VOS_END: u8 = 0xB1;
/// `user_data_start_code`.
pub(crate) const USER_DATA: u8 = 0xB2;
/// `group_of_vop_start_code`.
pub(crate) const GOV_START: u8 = 0xB3;
/// `visual_object_start_code`.
pub(crate) const VISUAL_OBJECT: u8 = 0xB5;
/// `vop_start_code`.
pub(crate) const VOP_START: u8 = 0xB6;

/// The largest frame edge a VOL may declare: the field is 13 bits.
pub(crate) const MAX_EDGE: u32 = 8191;

/// `video_object_type_indication` of the Simple Studio and Core Studio
/// objects, which use a different syntax.
const STUDIO_TYPES: [u8; 2] = [0x0E, 0x0F];

/// One start-code delimited unit of a stream: the byte after `00 00 01`
/// and the payload up to the next start code.
pub(crate) struct Unit<'a> {
    pub(crate) code: u8,
    pub(crate) payload: &'a [u8],
}

/// Splits `data` at `00 00 01` start codes. Bytes before the first start
/// code are dropped. Coded VOP data never holds 23 zero bits in a row, so
/// a start code cannot appear inside one.
pub(crate) fn units(data: &[u8]) -> Vec<Unit<'_>> {
    let mut starts = Vec::new();
    let mut i = 0;
    while let Some(window) = data.get(i..i + 3) {
        if window == [0, 0, 1] {
            starts.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (n, &start) in starts.iter().enumerate() {
        let end = starts.get(n + 1).copied().unwrap_or(data.len());
        if let (Some(&code), Some(payload)) = (data.get(start + 3), data.get(start + 4..end)) {
            out.push(Unit { code, payload });
        }
    }
    out
}

/// Whether `data` opens with the 22-bit `short_video_start_marker`.
pub(crate) fn is_short_header(data: &[u8]) -> bool {
    matches!(data, [0, 0, b, ..] if b & 0xFC == 0x80)
}

/// Whether a start code byte opens a video object (`0x00`–`0x1F`).
pub(crate) fn is_video_object(code: u8) -> bool {
    code <= 0x1F
}

/// Whether a start code byte opens a video object layer (`0x20`–`0x2F`).
pub(crate) fn is_vol(code: u8) -> bool {
    (0x20..=0x2F).contains(&code)
}

/// Colour signalled by `video_signal_type` (Section 6.3.2), as the
/// ISO/IEC 23091-2 code points.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Colour {
    pub(crate) full_range: bool,
    pub(crate) primaries: Option<u8>,
    pub(crate) transfer: Option<u8>,
    pub(crate) matrix: Option<u8>,
}

/// `visual_object_sequence`'s profile and level indication.
pub(crate) fn parse_vos(payload: &[u8]) -> Option<u8> {
    payload.first().copied()
}

/// The colour description of a `visual_object` header, when it names a
/// video object.
pub(crate) fn parse_visual_object(payload: &[u8]) -> Colour {
    let mut r = BitReader::new(payload);
    if r.bit() {
        r.skip(4 + 3); // verid, priority
    }
    let kind = r.read(4);
    let mut colour = Colour::default();
    // Video ID 1 and still texture ID 2 carry video_signal_type.
    if (kind == 1 || kind == 2) && r.bit() {
        r.skip(3); // video_format
        colour.full_range = r.bit();
        if r.bit() {
            colour.primaries = Some(r.read(8) as u8);
            colour.transfer = Some(r.read(8) as u8);
            colour.matrix = Some(r.read(8) as u8);
        }
    }
    if r.overrun() {
        Colour::default()
    } else {
        colour
    }
}

/// `video_object_layer_sprite_usage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpriteUsage {
    None,
    Static,
    Gmc,
}

/// A decoded `VideoObjectLayer()` header.
#[derive(Clone, Debug)]
pub(crate) struct Vol {
    pub(crate) vo_type: u8,
    pub(crate) verid: u8,
    /// Pixel aspect ratio, width over height; `(0, 0)` when unspecified.
    pub(crate) pixel_aspect: (u8, u8),
    pub(crate) vol_control: bool,
    pub(crate) low_delay: bool,
    pub(crate) time_resolution: u32,
    pub(crate) time_bits: u32,
    pub(crate) fixed_increment: Option<u32>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) interlaced: bool,
    pub(crate) sprite: SpriteUsage,
    pub(crate) warping_points: u8,
    pub(crate) warping_accuracy: u8,
    pub(crate) quant_precision: u32,
    pub(crate) mpeg_quant: bool,
    /// Raster order.
    pub(crate) intra_matrix: [u8; 64],
    /// Raster order.
    pub(crate) inter_matrix: [u8; 64],
    pub(crate) quarter_sample: bool,
    /// Bits of complexity estimation every VOP header carries, by I, P
    /// and B VOP.
    pub(crate) estimation_bits: [u32; 3],
    pub(crate) resync: bool,
    pub(crate) data_partitioning: bool,
    pub(crate) rvlc: bool,
}

/// Why a VOL is outside what this decoder implements.
fn unsupported(what: &'static str) -> CodecError {
    CodecError::Unsupported { what }
}

/// Reads a custom quantiser matrix in zigzag order. A zero ends it early
/// and the last value repeats (Section 6.3.3).
fn read_matrix(r: &mut BitReader<'_>) -> Result<[u8; 64], CodecError> {
    let mut m = [0u8; 64];
    let mut last = 0u8;
    let mut i = 0;
    while i < 64 {
        let v = r.read(8) as u8;
        if v == 0 {
            break;
        }
        last = v;
        m[usize::from(ZIGZAG[i])] = v;
        i += 1;
    }
    if i == 0 {
        return Err(r.malformed("quantiser matrix starts with zero"));
    }
    while i < 64 {
        m[usize::from(ZIGZAG[i])] = last;
        i += 1;
    }
    Ok(m)
}

/// Skips `define_vop_complexity_estimation_header` and returns the bits
/// each VOP header then carries, by I, P and B VOP.
fn read_estimation(r: &mut BitReader<'_>) -> Result<[u32; 3], CodecError> {
    let method = r.read(2);
    if method >= 2 {
        return Err(unsupported("complexity estimation method"));
    }
    let mut bits = [0u32; 3];
    let mut flags = |r: &mut BitReader<'_>, weights: &[(usize, u32)]| {
        if !r.bit() {
            for &(kind, w) in weights {
                if r.bit() {
                    bits[kind] += w;
                }
            }
        }
    };
    // shape_complexity_estimation_disable: opaque, transparent, intra_cae,
    // inter_cae, no_update, upsampling.
    flags(r, &[(0, 8), (0, 8), (0, 8), (0, 8), (0, 8), (0, 8)]);
    // texture_complexity_estimation_set_1_disable: intra, inter, inter4v,
    // not_coded blocks.
    flags(r, &[(0, 8), (1, 8), (1, 8), (0, 8)]);
    r.marker("complexity estimation")?;
    // set 2: dct_coefs, dct_lines, vlc_symbols, vlc_bits.
    flags(r, &[(0, 8), (0, 8), (0, 8), (0, 4)]);
    // motion_compensation_complexity_disable: apm, npm,
    // interpolate_mc_q, forw_back_mc_q, halfpel2, halfpel4.
    flags(r, &[(1, 8), (1, 8), (2, 8), (1, 8), (1, 8), (1, 8)]);
    r.marker("complexity estimation")?;
    if method == 1 {
        // version2_complexity_estimation_disable: sadct, quarterpel.
        flags(r, &[(0, 8), (1, 8)]);
    }
    Ok(bits)
}

/// Parses a `VideoObjectLayer()` payload (after its start code).
pub(crate) fn parse_vol(payload: &[u8]) -> Result<Vol, CodecError> {
    let mut r = BitReader::new(payload);
    r.skip(1); // random_accessible_vol
    let vo_type = r.read(8) as u8;
    if STUDIO_TYPES.contains(&vo_type) {
        return Err(unsupported("studio profile"));
    }
    let verid = if r.bit() {
        let verid = r.read(4) as u8;
        r.skip(3); // priority
        verid
    } else {
        1
    };
    let aspect = r.read(4);
    let pixel_aspect = match aspect {
        1 => (1, 1),
        2 => (12, 11),
        3 => (10, 11),
        4 => (16, 11),
        5 => (40, 33),
        15 => (r.read(8) as u8, r.read(8) as u8),
        _ => (0, 0),
    };
    let vol_control = r.bit();
    // Absent `vol_control_parameters`, encoders that code B-VOPs still
    // leave the flag unsent, so only a sent flag is trusted.
    let mut low_delay = false;
    if vol_control {
        let chroma = r.read(2);
        if chroma != 1 {
            return Err(unsupported("chroma format other than 4:2:0"));
        }
        low_delay = r.bit();
        if r.bit() {
            // vbv_parameters: 79 bits with their markers.
            r.skip(15);
            r.marker("vbv")?;
            r.skip(15);
            r.marker("vbv")?;
            r.skip(15);
            r.marker("vbv")?;
            r.skip(3 + 11);
            r.marker("vbv")?;
            r.skip(15);
            r.marker("vbv")?;
        }
    }
    let shape = r.read(2);
    if shape != 0 {
        return Err(unsupported("arbitrary shape"));
    }
    // Some encoders leave the markers clear; the fields still parse.
    r.skip(1);
    let time_resolution = r.read(16);
    if time_resolution == 0 {
        return Err(r.malformed("vop_time_increment_resolution is zero"));
    }
    let time_bits = (32 - (time_resolution - 1).leading_zeros()).max(1);
    r.skip(1);
    let fixed_increment = if r.bit() {
        Some(r.read(time_bits))
    } else {
        None
    };
    r.skip(1);
    let width = r.read(13);
    r.skip(1);
    let height = r.read(13);
    r.skip(1);
    if width == 0 || height == 0 {
        return Err(r.malformed("zero frame size"));
    }
    let interlaced = r.bit();
    // obmc_disable: encoders that clear it do not code OBMC either.
    r.skip(1);
    let sprite = match if verid == 1 { r.read(1) } else { r.read(2) } {
        0 => SpriteUsage::None,
        1 => SpriteUsage::Static,
        2 => SpriteUsage::Gmc,
        _ => return Err(r.malformed("reserved sprite usage")),
    };
    if sprite == SpriteUsage::Static {
        return Err(unsupported("static sprite"));
    }
    let (mut warping_points, mut warping_accuracy) = (0, 0);
    if sprite == SpriteUsage::Gmc {
        warping_points = r.read(6) as u8;
        if warping_points > 3 {
            return Err(unsupported("more than three GMC warping points"));
        }
        warping_accuracy = r.read(2) as u8;
        if r.bit() {
            return Err(unsupported("sprite brightness change"));
        }
    }
    let mut quant_precision = 5;
    if r.bit() {
        quant_precision = r.read(4);
        let bits_per_pixel = r.read(4);
        if bits_per_pixel != 8 || quant_precision != 5 {
            return Err(unsupported("N-bit video"));
        }
    }
    let mpeg_quant = r.bit();
    let mut intra_matrix = DEFAULT_INTRA_MATRIX;
    let mut inter_matrix = DEFAULT_INTER_MATRIX;
    if mpeg_quant {
        if r.bit() {
            intra_matrix = read_matrix(&mut r)?;
        }
        if r.bit() {
            inter_matrix = read_matrix(&mut r)?;
        }
    }
    let quarter_sample = verid != 1 && r.bit();
    let estimation_bits = if r.bit() {
        [0; 3]
    } else {
        read_estimation(&mut r)?
    };
    let resync = !r.bit();
    let data_partitioning = r.bit();
    let rvlc = data_partitioning && r.bit();
    if verid != 1 {
        if r.bit() {
            return Err(unsupported("NEWPRED"));
        }
        if r.bit() {
            return Err(unsupported("reduced resolution VOP"));
        }
    }
    if r.bit() {
        return Err(unsupported("scalability"));
    }
    r.check("video object layer")?;
    if data_partitioning && interlaced {
        return Err(unsupported("interlaced data partitioning"));
    }
    Ok(Vol {
        vo_type,
        verid,
        pixel_aspect,
        vol_control,
        low_delay,
        time_resolution,
        time_bits,
        fixed_increment,
        width,
        height,
        interlaced,
        sprite,
        warping_points,
        warping_accuracy,
        quant_precision,
        mpeg_quant,
        intra_matrix,
        inter_matrix,
        quarter_sample,
        estimation_bits,
        resync,
        data_partitioning,
        rvlc,
    })
}

/// `vop_coding_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VopType {
    I,
    P,
    B,
    S,
}

/// A decoded `VideoObjectPlane()` header.
#[derive(Clone, Debug)]
pub(crate) struct Vop {
    pub(crate) kind: VopType,
    pub(crate) modulo_time_base: u32,
    pub(crate) time_increment: u32,
    pub(crate) coded: bool,
    pub(crate) rounding: bool,
    pub(crate) dc_threshold: u32,
    pub(crate) top_field_first: bool,
    pub(crate) alternate_scan: bool,
    /// Sprite trajectory differences, one per warping point.
    pub(crate) trajectory: [(i32, i32); 3],
    pub(crate) quant: u32,
    pub(crate) fcode_forward: u32,
    pub(crate) fcode_backward: u32,
    /// Bit offset of the first macroblock in the payload.
    pub(crate) data_start: usize,
}

/// Reads one sprite trajectory component (Section 6.3.5.4).
fn read_trajectory_component(r: &mut BitReader<'_>) -> Result<i32, CodecError> {
    let Some(len) = tables().dmv_length.read(r) else {
        return Err(r.malformed("dmv_length"));
    };
    Ok(r.read_xbits(u32::from(len)))
}

/// Reads `sprite_trajectory()`.
pub(crate) fn read_trajectory(
    r: &mut BitReader<'_>,
    points: u8,
    lenient_first_marker: bool,
) -> Result<[(i32, i32); 3], CodecError> {
    let mut out = [(0, 0); 3];
    for slot in out.iter_mut().take(usize::from(points)) {
        let x = read_trajectory_component(r)?;
        if lenient_first_marker {
            // One DivX 5 build omits this marker.
            if r.peek(1) == 1 {
                r.skip(1);
            }
        } else {
            r.marker("sprite trajectory")?;
        }
        let y = read_trajectory_component(r)?;
        r.marker("sprite trajectory")?;
        *slot = (x, y);
    }
    Ok(out)
}

/// Parses a `VideoObjectPlane()` payload against its VOL.
pub(crate) fn parse_vop(payload: &[u8], vol: &Vol, divx_413: bool) -> Result<Vop, CodecError> {
    let mut r = BitReader::new(payload);
    let kind = match r.read(2) {
        0 => VopType::I,
        1 => VopType::P,
        2 => VopType::B,
        _ => VopType::S,
    };
    let mut modulo_time_base = 0;
    while r.bit() {
        modulo_time_base += 1;
        if r.overrun() || modulo_time_base > 3600 {
            return Err(r.malformed("modulo_time_base"));
        }
    }
    r.skip(1); // marker
    let time_increment = r.read(vol.time_bits);
    r.skip(1); // marker
    let coded = r.bit();
    let mut vop = Vop {
        kind,
        modulo_time_base,
        time_increment,
        coded,
        rounding: false,
        dc_threshold: 0,
        top_field_first: false,
        alternate_scan: false,
        trajectory: [(0, 0); 3],
        quant: 1,
        fcode_forward: 1,
        fcode_backward: 1,
        data_start: 0,
    };
    if !coded {
        r.check("vop header")?;
        return Ok(vop);
    }
    if kind == VopType::S && vol.sprite != SpriteUsage::Gmc {
        return Err(r.malformed("S-VOP without GMC"));
    }
    if kind == VopType::P || kind == VopType::S {
        vop.rounding = r.bit();
    }
    let estimation = match kind {
        VopType::I => vol.estimation_bits[0],
        VopType::P | VopType::S => vol.estimation_bits[0] + vol.estimation_bits[1],
        VopType::B => vol.estimation_bits.iter().sum(),
    };
    r.skip(estimation);
    vop.dc_threshold = INTRA_DC_THRESHOLD[r.read(3) as usize & 7];
    if vol.interlaced {
        vop.top_field_first = r.bit();
        vop.alternate_scan = r.bit();
    }
    if kind == VopType::S {
        vop.trajectory = read_trajectory(&mut r, vol.warping_points, divx_413)?;
    }
    vop.quant = r.read(vol.quant_precision);
    if vop.quant == 0 {
        return Err(r.malformed("vop_quant is zero"));
    }
    if kind != VopType::I {
        vop.fcode_forward = r.read(3);
        if vop.fcode_forward == 0 {
            return Err(r.malformed("vop_fcode_forward is zero"));
        }
    }
    if kind == VopType::B {
        vop.fcode_backward = r.read(3);
        if vop.fcode_backward == 0 {
            return Err(r.malformed("vop_fcode_backward is zero"));
        }
    }
    r.check("vop header")?;
    vop.data_start = r.position();
    Ok(vop)
}

/// Seconds of a `group_of_vop` time code.
pub(crate) fn parse_gov(payload: &[u8]) -> Option<u32> {
    let mut r = BitReader::new(payload);
    let hours = r.read(5);
    let minutes = r.read(6);
    r.skip(1);
    let seconds = r.read(6);
    if r.overrun() {
        None
    } else {
        Some(seconds + 60 * (minutes + 60 * hours))
    }
}

/// The encoder a stream's user data names, for the decoding differences
/// some versions need.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Encoder {
    pub(crate) divx_version: Option<u32>,
    pub(crate) divx_build: Option<u32>,
    /// DivX "packed bitstream": a B-VOP shares its P-VOP's packet.
    pub(crate) divx_packed: bool,
    pub(crate) xvid_build: Option<u32>,
    pub(crate) lavc_build: Option<u32>,
}

/// Leading decimal digits of `s`, and the rest.
fn number(s: &str) -> Option<(u32, &str)> {
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let n = s.get(..end)?.parse().ok()?;
    Some((n, s.get(end..)?))
}

/// Updates `enc` from one user data payload.
pub(crate) fn parse_user_data(payload: &[u8], enc: &mut Encoder) {
    let end = payload
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(payload.len());
    let text = String::from_utf8_lossy(payload.get(..end.min(255)).unwrap_or_default());
    let text = text.as_ref();
    if let Some(rest) = text.strip_prefix("DivX")
        && let Some((version, rest)) = number(rest)
    {
        let rest = rest
            .strip_prefix("Build")
            .or_else(|| rest.strip_prefix('b'));
        if let Some(rest) = rest
            && let Some((build, tail)) = number(rest)
        {
            enc.divx_version = Some(version);
            enc.divx_build = Some(build);
            enc.divx_packed = tail.starts_with('p');
        }
    }
    if let Some(rest) = text.strip_prefix("XviD")
        && let Some((build, _)) = number(rest)
    {
        enc.xvid_build = Some(build);
    }
    if let Some(rest) = text.strip_prefix("Lavc") {
        let mut parts = rest.split('.');
        let mut v = [0u32; 3];
        let mut ok = true;
        for slot in &mut v {
            match parts.next().and_then(|p| number(p).map(|n| n.0)) {
                Some(n) => *slot = n & 0xFF,
                None => ok = false,
            }
        }
        if ok {
            enc.lavc_build = Some((v[0] << 16) + (v[1] << 8) + v[2]);
        }
    } else if let Some(pos) = text.find("libavcodec build: ") {
        if let Some((build, _)) = text.get(pos + 18..).and_then(number) {
            enc.lavc_build = Some(build);
        }
    } else if text.starts_with("FFmpe")
        && let Some(pos) = text.find('b')
        && let Some((build, _)) = text.get(pos + 1..).and_then(number)
    {
        enc.lavc_build = Some(build);
    } else if text == "ffmpeg" {
        enc.lavc_build = Some(4600);
    }
    if enc.xvid_build.is_some() {
        enc.divx_version = None;
        enc.divx_build = None;
    }
}

/// A short-header (H.263 baseline) picture header.
#[derive(Clone, Debug)]
pub(crate) struct ShortHeader {
    pub(crate) temporal_reference: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) inter: bool,
    pub(crate) quant: u32,
    pub(crate) gobs: u32,
    /// Macroblock rows in one group of blocks.
    pub(crate) gob_rows: u32,
    pub(crate) data_start: usize,
}

/// Parses `video_plane_with_short_header()` up to the first macroblock.
pub(crate) fn parse_short_header(payload: &[u8]) -> Result<ShortHeader, CodecError> {
    let mut r = BitReader::new(payload);
    if r.read(22) != 0x20 {
        return Err(r.malformed("short_video_start_marker"));
    }
    let temporal_reference = r.read(8);
    r.marker("short header")?;
    if r.bit() {
        return Err(r.malformed("short header zero bit"));
    }
    r.skip(3); // split_screen, document_camera, full_picture_freeze
    let (width, height, gobs, gob_rows) = match r.read(3) {
        1 => (128, 96, 6, 1),
        2 => (176, 144, 9, 1),
        3 => (352, 288, 18, 1),
        4 => (704, 576, 18, 2),
        5 => (1408, 1152, 18, 4),
        _ => return Err(unsupported("short header source format")),
    };
    let inter = r.bit();
    if r.read(4) != 0 {
        return Err(unsupported("H.263 optional modes"));
    }
    let quant = r.read(5);
    if quant == 0 {
        return Err(r.malformed("vop_quant is zero"));
    }
    r.skip(1); // zero_bit
    let mut extra = 0;
    while r.bit() {
        r.skip(8); // psupp
        extra += 1;
        if r.overrun() || extra > 256 {
            return Err(r.malformed("pei"));
        }
    }
    r.check("short header")?;
    Ok(ShortHeader {
        temporal_reference,
        width,
        height,
        inter,
        quant,
        gobs,
        gob_rows,
        data_start: r.position(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Encoder, SpriteUsage, is_short_header, parse_short_header, parse_user_data, parse_vol,
        units,
    };
    use crate::bits::BitWriter;

    /// A Simple profile VOL: 30000 ticks a second, 176 by 144, progressive.
    pub(crate) fn simple_vol(width: u32, height: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.put(1, 0); // random_accessible_vol
        w.put(8, 1); // simple object
        w.put(1, 0); // is_object_layer_identifier
        w.put(4, 1); // square pixels
        w.put(1, 0); // vol_control_parameters
        w.put(2, 0); // rectangular
        w.put(1, 1);
        w.put(16, 30000);
        w.put(1, 1);
        w.put(1, 0); // fixed_vop_rate
        w.put(1, 1);
        w.put(13, width);
        w.put(1, 1);
        w.put(13, height);
        w.put(1, 1);
        w.put(1, 0); // interlaced
        w.put(1, 1); // obmc_disable
        w.put(1, 0); // sprite
        w.put(1, 0); // not_8_bit
        w.put(1, 0); // quant_type
        w.put(1, 1); // complexity_estimation_disable
        w.put(1, 1); // resync_marker_disable
        w.put(1, 0); // data_partitioned
        w.put(1, 0); // scalability
        w.stuff();
        w.finish()
    }

    #[test]
    fn parses_a_simple_vol() {
        let vol = parse_vol(&simple_vol(176, 144)).expect("vol");
        assert_eq!((vol.width, vol.height), (176, 144));
        assert_eq!(vol.time_resolution, 30000);
        assert_eq!(vol.time_bits, 15);
        assert!(
            !vol.vol_control && !vol.low_delay,
            "an unsent flag is not low delay"
        );
        assert!(!vol.resync && !vol.mpeg_quant && !vol.quarter_sample);
        assert_eq!(vol.sprite, SpriteUsage::None);
    }

    #[test]
    fn refuses_a_truncated_vol_and_a_zero_size() {
        let bytes = simple_vol(176, 144);
        assert!(parse_vol(&bytes[..4]).is_err());
        assert!(parse_vol(&simple_vol(0, 144)).is_err());
    }

    #[test]
    fn splits_start_codes_but_not_resync_markers() {
        // 00 00 80 inside a VOP is a resync marker, not a start code.
        let data = [
            0xFF, 0, 0, 1, 0xB0, 0x01, 0, 0, 1, 0xB6, 0x10, 0x20, 0, 0, 0x80, 0x02,
        ];
        let u = units(&data);
        assert_eq!(u.len(), 2);
        assert_eq!((u[0].code, u[0].payload), (0xB0, &[0x01][..]));
        assert_eq!(u[1].code, 0xB6);
        assert_eq!(u[1].payload, &[0x10, 0x20, 0, 0, 0x80, 0x02][..]);
        assert!(is_short_header(&[0, 0, 0x82, 0]));
        assert!(!is_short_header(&data));
    }

    #[test]
    fn reads_encoder_names_from_user_data() {
        let mut enc = Encoder::default();
        parse_user_data(b"DivX503b1393p", &mut enc);
        assert_eq!(enc.divx_version, Some(503));
        assert_eq!(enc.divx_build, Some(1393));
        assert!(enc.divx_packed);
        let mut enc = Encoder::default();
        parse_user_data(b"Lavc58.54.100", &mut enc);
        assert_eq!(enc.lavc_build, Some((58 << 16) + (54 << 8) + 100));
        let mut enc = Encoder::default();
        parse_user_data(b"XviD0050", &mut enc);
        assert_eq!(enc.xvid_build, Some(50));
    }

    #[test]
    fn parses_a_qcif_short_header() {
        let mut w = BitWriter::new();
        w.put(22, 0x20);
        w.put(8, 7); // temporal reference
        w.put(1, 1);
        w.put(1, 0);
        w.put(3, 0);
        w.put(3, 2); // QCIF
        w.put(1, 1); // inter
        w.put(4, 0);
        w.put(5, 12);
        w.put(1, 0);
        w.put(1, 0); // pei
        let bytes = w.finish();
        let h = parse_short_header(&bytes).expect("header");
        assert_eq!((h.width, h.height, h.gobs, h.quant), (176, 144, 9, 12));
        assert!(h.inter);
        assert_eq!(h.temporal_reference, 7);
    }
}
