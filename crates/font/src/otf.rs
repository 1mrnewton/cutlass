//! OpenType outlines stored in a `CFF ` table.
//!
//! Glyphs are Type 2 charstrings. Advances still come from `hmtx`; the
//! charstring width is only consumed so the path operators stay aligned.

use crate::{Contour, Directory, FontError, FontKind, Outline, Segment, bytes_at, i16_at, u16_at};

const STACK_LIMIT: usize = 48;
const HINT_LIMIT: usize = 96;
const SUBR_DEPTH: u32 = 10;
const TRANSIENT: usize = 32;

pub(crate) struct Source<'a> {
    kind: Kind<'a>,
}

enum Kind<'a> {
    Cff(Cff<'a>),
    Unsupported,
}

struct Cff<'a> {
    charstrings: Index<'a>,
    gsubrs: Index<'a>,
    subrs: Index<'a>,
}

struct Index<'a> {
    data: &'a [u8],
    count: usize,
    off_size: usize,
    offset_at: usize,
    data_start: usize,
}

struct Dict {
    charstrings: Option<usize>,
    private_size: Option<usize>,
    private_at: Option<usize>,
    subrs: Option<usize>,
    charstring_type: i32,
    cid: bool,
}

struct Pen {
    contours: Vec<Contour>,
    segments: Vec<Segment>,
    x: f32,
    y: f32,
    start_x: f32,
    start_y: f32,
    open: bool,
    width_seen: bool,
    hint_count: usize,
    stack: Vec<f32>,
    transient: [f32; TRANSIENT],
}

enum Stop {
    End,
    Return,
}

pub(crate) fn source<'a>(directory: &Directory<'a>) -> Result<Source<'a>, FontError> {
    if directory.kind() != FontKind::OpenType {
        return Err(FontError::BadTable);
    }
    match directory.table(b"CFF ") {
        Ok(table) => Ok(Source {
            kind: Kind::Cff(parse_cff(table)?),
        }),
        Err(FontError::MissingTable) if directory.table(b"CFF2").is_ok() => Ok(Source {
            kind: Kind::Unsupported,
        }),
        Err(error) => Err(error),
    }
}

pub(crate) fn has_outlines(source: &Source<'_>) -> bool {
    matches!(source.kind, Kind::Cff(_))
}

pub(crate) fn outline(source: &Source<'_>, glyph_id: u16) -> Result<Outline, FontError> {
    let Kind::Cff(cff) = &source.kind else {
        return Err(FontError::Unsupported);
    };
    let program = cff.charstrings.get(usize::from(glyph_id))?;
    let mut pen = Pen::new();
    match interpret(program, cff, &mut pen, 0)? {
        Stop::End => Ok(pen.outline()),
        Stop::Return => Err(FontError::BadTable),
    }
}

fn parse_cff(data: &[u8]) -> Result<Cff<'_>, FontError> {
    if data.len() < 4 {
        return Err(FontError::Truncated);
    }
    if data[0] != 1 {
        return Err(FontError::Unsupported);
    }
    let header = usize::from(data[2]);
    if header < 4 {
        return Err(FontError::BadTable);
    }
    let mut pos = header;
    let names = Index::parse(data, &mut pos)?;
    if names.count == 0 {
        return Err(FontError::EmptyTable);
    }
    let tops = Index::parse(data, &mut pos)?;
    if tops.count != names.count {
        return Err(FontError::BadTable);
    }
    let _strings = Index::parse(data, &mut pos)?;
    let gsubrs = Index::parse(data, &mut pos)?;
    let top = parse_dict(tops.get(0)?)?;
    if top.cid || top.charstring_type != 2 {
        return Err(FontError::Unsupported);
    }
    let Some(charstrings_at) = top.charstrings else {
        return Err(FontError::BadTable);
    };
    let mut cursor = charstrings_at;
    let charstrings = Index::parse(data, &mut cursor)?;
    if charstrings.count == 0 {
        return Err(FontError::EmptyTable);
    }
    let subrs = private_subrs(data, &top)?;
    Ok(Cff {
        charstrings,
        gsubrs,
        subrs,
    })
}

fn private_subrs<'a>(data: &'a [u8], top: &Dict) -> Result<Index<'a>, FontError> {
    let (Some(size), Some(at)) = (top.private_size, top.private_at) else {
        return Ok(Index::empty());
    };
    if size == 0 {
        return Ok(Index::empty());
    }
    let bytes = bytes_at(data, at, size)?;
    let private = parse_dict(bytes)?;
    if let Some(relative) = private.subrs {
        let mut cursor = at.checked_add(relative).ok_or(FontError::BadTable)?;
        Index::parse(data, &mut cursor)
    } else {
        Ok(Index::empty())
    }
}

impl<'a> Index<'a> {
    fn empty() -> Self {
        Self {
            data: &[],
            count: 0,
            off_size: 1,
            offset_at: 0,
            data_start: 0,
        }
    }

    fn parse(data: &'a [u8], pos: &mut usize) -> Result<Self, FontError> {
        let count = usize::from(u16_at(data, *pos)?);
        *pos += 2;
        if count == 0 {
            return Ok(Self::empty());
        }
        let off_size = usize::from(*data.get(*pos).ok_or(FontError::Truncated)?);
        if !(1..=4).contains(&off_size) {
            return Err(FontError::BadTable);
        }
        *pos += 1;
        let offset_at = *pos;
        let width = count
            .checked_add(1)
            .and_then(|entries| entries.checked_mul(off_size))
            .ok_or(FontError::BadTable)?;
        let offsets = bytes_at(data, offset_at, width)?;
        *pos += width;
        let data_start = *pos;
        let mut previous = 0usize;
        for index in 0..=count {
            let offset = read_offset(offsets, index, off_size)?;
            if offset == 0 || offset < previous {
                return Err(FontError::BadTable);
            }
            previous = offset;
        }
        let end = data_start
            .checked_add(previous - 1)
            .ok_or(FontError::BadTable)?;
        if end > data.len() {
            return Err(FontError::Truncated);
        }
        *pos = end;
        Ok(Self {
            data,
            count,
            off_size,
            offset_at,
            data_start,
        })
    }

    fn get(&self, index: usize) -> Result<&'a [u8], FontError> {
        if index >= self.count {
            return Err(FontError::BadTable);
        }
        let start = self.offset_value(index)?;
        let end = self.offset_value(index + 1)?;
        if end < start {
            return Err(FontError::BadTable);
        }
        let from = self
            .data_start
            .checked_add(start - 1)
            .ok_or(FontError::BadTable)?;
        let to = self
            .data_start
            .checked_add(end - 1)
            .ok_or(FontError::BadTable)?;
        bytes_at(self.data, from, to - from)
    }

    fn offset_value(&self, index: usize) -> Result<usize, FontError> {
        let offsets = bytes_at(
            self.data,
            self.offset_at,
            (self.count + 1)
                .checked_mul(self.off_size)
                .ok_or(FontError::BadTable)?,
        )?;
        read_offset(offsets, index, self.off_size)
    }
}

fn read_offset(data: &[u8], index: usize, off_size: usize) -> Result<usize, FontError> {
    let at = index.checked_mul(off_size).ok_or(FontError::BadTable)?;
    let bytes = bytes_at(data, at, off_size)?;
    let mut value = 0usize;
    for byte in bytes {
        value = value
            .checked_mul(256)
            .and_then(|value| value.checked_add(usize::from(*byte)))
            .ok_or(FontError::BadTable)?;
    }
    Ok(value)
}

fn parse_dict(data: &[u8]) -> Result<Dict, FontError> {
    let mut dict = Dict {
        charstrings: None,
        private_size: None,
        private_at: None,
        subrs: None,
        charstring_type: 2,
        cid: false,
    };
    let mut pos = 0;
    let mut stack = Vec::new();
    while pos < data.len() {
        let byte = data[pos];
        if dict_number(byte) {
            let (value, used) = read_dict_number(&data[pos..])?;
            pos += used;
            if stack.len() >= STACK_LIMIT {
                return Err(FontError::BadTable);
            }
            stack.push(value);
            continue;
        }
        if byte == 12 {
            let operator = *data.get(pos + 1).ok_or(FontError::Truncated)?;
            pos += 2;
            apply_escape(&mut dict, operator, &stack)?;
        } else if byte <= 21 {
            pos += 1;
            apply_operator(&mut dict, byte, &stack)?;
        } else {
            return Err(FontError::BadTable);
        }
        stack.clear();
    }
    Ok(dict)
}

fn dict_number(byte: u8) -> bool {
    matches!(byte, 28 | 29 | 30 | 32..=254)
}

fn apply_operator(dict: &mut Dict, operator: u8, stack: &[f64]) -> Result<(), FontError> {
    match operator {
        17 => dict.charstrings = Some(one_offset(stack)?),
        18 => {
            let (size, at) = two_offsets(stack)?;
            dict.private_size = Some(size);
            dict.private_at = Some(at);
        }
        19 => dict.subrs = Some(one_offset(stack)?),
        _ => {}
    }
    Ok(())
}

fn apply_escape(dict: &mut Dict, operator: u8, stack: &[f64]) -> Result<(), FontError> {
    match operator {
        6 => {
            dict.charstring_type = if one_number(stack)? == 2.0 { 2 } else { -1 };
        }
        30 => dict.cid = true,
        _ => {}
    }
    Ok(())
}

fn one_offset(stack: &[f64]) -> Result<usize, FontError> {
    let value = stack.first().copied().ok_or(FontError::BadTable)?;
    as_usize(value)
}

fn two_offsets(stack: &[f64]) -> Result<(usize, usize), FontError> {
    if stack.len() < 2 {
        return Err(FontError::BadTable);
    }
    Ok((as_usize(stack[0])?, as_usize(stack[1])?))
}

fn one_number(stack: &[f64]) -> Result<f32, FontError> {
    let value = stack.first().copied().ok_or(FontError::BadTable)?;
    let number = value as f32;
    if !number.is_finite() {
        return Err(FontError::BadTable);
    }
    Ok(number)
}

fn as_usize(value: f64) -> Result<usize, FontError> {
    if !(0.0..=f64::from(u32::MAX)).contains(&value) {
        return Err(FontError::BadTable);
    }
    let integer = value as u32;
    if f64::from(integer) != value {
        return Err(FontError::BadTable);
    }
    usize::try_from(integer).map_err(|_| FontError::BadTable)
}

fn read_dict_number(data: &[u8]) -> Result<(f64, usize), FontError> {
    let Some(first) = data.first().copied() else {
        return Err(FontError::Truncated);
    };
    match first {
        28 => {
            if data.len() < 3 {
                return Err(FontError::Truncated);
            }
            Ok((f64::from(i16_at(data, 1)?), 3))
        }
        29 => {
            if data.len() < 5 {
                return Err(FontError::Truncated);
            }
            let value = i32::from_be_bytes([data[1], data[2], data[3], data[4]]);
            Ok((f64::from(value), 5))
        }
        30 => read_real(data),
        32..=246 => Ok((f64::from(i16::from(first) - 139), 1)),
        247..=250 => {
            let second = *data.get(1).ok_or(FontError::Truncated)?;
            let value = (i32::from(first) - 247) * 256 + i32::from(second) + 108;
            Ok((f64::from(value), 2))
        }
        251..=254 => {
            let second = *data.get(1).ok_or(FontError::Truncated)?;
            let value = -((i32::from(first) - 251) * 256) - i32::from(second) - 108;
            Ok((f64::from(value), 2))
        }
        _ => Err(FontError::BadTable),
    }
}

fn read_real(data: &[u8]) -> Result<(f64, usize), FontError> {
    let mut pos = 1;
    let mut digits = String::new();
    loop {
        let byte = *data.get(pos).ok_or(FontError::Truncated)?;
        pos += 1;
        for nibble in [byte >> 4, byte & 0x0f] {
            match nibble {
                0..=9 => digits.push(char::from(b'0' + nibble)),
                0x0a => digits.push('.'),
                0x0b => digits.push('E'),
                0x0c => digits.push_str("E-"),
                0x0e if digits.is_empty() => digits.push('-'),
                0x0f => {
                    let value = digits.parse::<f64>().map_err(|_| FontError::BadTable)?;
                    if !value.is_finite() {
                        return Err(FontError::BadTable);
                    }
                    return Ok((value, pos));
                }
                _ => return Err(FontError::BadTable),
            }
            if digits.len() > 64 {
                return Err(FontError::BadTable);
            }
        }
    }
}

impl Pen {
    fn new() -> Self {
        Self {
            contours: Vec::new(),
            segments: Vec::new(),
            x: 0.0,
            y: 0.0,
            start_x: 0.0,
            start_y: 0.0,
            open: false,
            width_seen: false,
            hint_count: 0,
            stack: Vec::new(),
            transient: [0.0; TRANSIENT],
        }
    }

    fn outline(mut self) -> Outline {
        self.close();
        Outline {
            contours: self.contours,
        }
    }

    fn close(&mut self) {
        if !self.open {
            return;
        }
        self.open = false;
        if self.segments.is_empty() {
            return;
        }
        let (end_x, end_y) = segment_end(self.segments.last());
        if end_x != self.start_x || end_y != self.start_y {
            self.segments.push(Segment::Line {
                x: self.start_x,
                y: self.start_y,
            });
        }
        self.contours.push(Contour {
            x: self.start_x,
            y: self.start_y,
            segments: std::mem::take(&mut self.segments),
        });
    }

    fn move_to(&mut self, dx: f32, dy: f32) -> Result<(), FontError> {
        self.close();
        self.x = finite(self.x + dx)?;
        self.y = finite(self.y + dy)?;
        self.start_x = self.x;
        self.start_y = self.y;
        self.open = true;
        Ok(())
    }

    fn line(&mut self, dx: f32, dy: f32) -> Result<(), FontError> {
        if !self.open {
            return Err(FontError::BadTable);
        }
        self.x = finite(self.x + dx)?;
        self.y = finite(self.y + dy)?;
        self.segments.push(Segment::Line {
            x: self.x,
            y: self.y,
        });
        Ok(())
    }

    fn curve(
        &mut self,
        dx1: f32,
        dy1: f32,
        dx2: f32,
        dy2: f32,
        dx3: f32,
        dy3: f32,
    ) -> Result<(), FontError> {
        if !self.open {
            return Err(FontError::BadTable);
        }
        let c1x = finite(self.x + dx1)?;
        let c1y = finite(self.y + dy1)?;
        let c2x = finite(c1x + dx2)?;
        let c2y = finite(c1y + dy2)?;
        self.x = finite(c2x + dx3)?;
        self.y = finite(c2y + dy3)?;
        self.segments.push(Segment::Cubic {
            c1x,
            c1y,
            c2x,
            c2y,
            x: self.x,
            y: self.y,
        });
        Ok(())
    }

    fn push(&mut self, value: f32) -> Result<(), FontError> {
        if !value.is_finite() || self.stack.len() >= STACK_LIMIT {
            return Err(FontError::BadTable);
        }
        self.stack.push(value);
        Ok(())
    }

    fn pop(&mut self) -> Result<f32, FontError> {
        self.stack.pop().ok_or(FontError::BadTable)
    }
}

fn segment_end(segment: Option<&Segment>) -> (f32, f32) {
    match segment {
        Some(Segment::Line { x, y } | Segment::Quad { x, y, .. } | Segment::Cubic { x, y, .. }) => {
            (*x, *y)
        }
        None => (0.0, 0.0),
    }
}

fn finite(value: f32) -> Result<f32, FontError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FontError::BadTable)
    }
}

fn interpret(bytes: &[u8], cff: &Cff<'_>, pen: &mut Pen, depth: u32) -> Result<Stop, FontError> {
    if depth > SUBR_DEPTH {
        return Err(FontError::BadTable);
    }
    let mut pos = 0;
    while pos < bytes.len() {
        let byte = bytes[pos];
        match byte {
            1 | 3 | 18 | 23 => stems(pen)?,
            19 | 20 => {
                if !pen.stack.is_empty() {
                    stems(pen)?;
                }
                pos += mask_length(pen, bytes.get(pos + 1..).ok_or(FontError::Truncated)?)?;
            }
            4 => vmoveto(pen)?,
            5 => rlineto(pen)?,
            6 => alternate_lines(pen, true)?,
            7 => alternate_lines(pen, false)?,
            8 => rrcurveto(pen)?,
            10 => match call_subr(cff, pen, depth, false)? {
                Stop::End => return Ok(Stop::End),
                Stop::Return => {}
            },
            11 => return Ok(Stop::Return),
            14 => {
                endchar(pen)?;
                return Ok(Stop::End);
            }
            21 => rmoveto(pen)?,
            22 => hmoveto(pen)?,
            24 => rcurveline(pen)?,
            25 => rlinecurve(pen)?,
            26 => vvcurveto(pen)?,
            27 => hhcurveto(pen)?,
            28 => {
                let value = f32::from(i16_at(bytes, pos + 1)?);
                pen.push(value)?;
                pos += 2;
            }
            29 => match call_subr(cff, pen, depth, true)? {
                Stop::End => return Ok(Stop::End),
                Stop::Return => {}
            },
            30 => hvcurveto(pen, false)?,
            31 => hvcurveto(pen, true)?,
            12 => {
                let operator = *bytes.get(pos + 1).ok_or(FontError::Truncated)?;
                pos += 1;
                escape(pen, operator)?;
            }
            32..=246 => pen.push(f32::from(i16::from(byte) - 139))?,
            247..=250 => {
                let second = *bytes.get(pos + 1).ok_or(FontError::Truncated)?;
                let value = (i32::from(byte) - 247) * 256 + i32::from(second) + 108;
                pen.push(value as f32)?;
                pos += 1;
            }
            251..=254 => {
                let second = *bytes.get(pos + 1).ok_or(FontError::Truncated)?;
                let value = -((i32::from(byte) - 251) * 256) - i32::from(second) - 108;
                pen.push(value as f32)?;
                pos += 1;
            }
            255 => {
                let raw = i32::from_be_bytes(
                    bytes
                        .get(pos + 1..pos + 5)
                        .ok_or(FontError::Truncated)?
                        .try_into()
                        .map_err(|_| FontError::Truncated)?,
                );
                pen.push(raw as f32 / 65_536.0)?;
                pos += 4;
            }
            _ => return Err(FontError::BadTable),
        }
        pos += 1;
    }
    Err(FontError::BadTable)
}

fn mask_length(pen: &Pen, rest: &[u8]) -> Result<usize, FontError> {
    if pen.hint_count == 0 || pen.hint_count > HINT_LIMIT {
        return Err(FontError::BadTable);
    }
    let length = pen.hint_count.div_ceil(8);
    if rest.len() < length {
        return Err(FontError::Truncated);
    }
    Ok(length)
}

fn stems(pen: &mut Pen) -> Result<(), FontError> {
    let args = take_width(pen, false)?;
    if !args.len().is_multiple_of(2) {
        return Err(FontError::BadTable);
    }
    pen.hint_count = pen
        .hint_count
        .checked_add(args.len() / 2)
        .ok_or(FontError::BadTable)?;
    if pen.hint_count > HINT_LIMIT {
        return Err(FontError::BadTable);
    }
    Ok(())
}

fn hmoveto(pen: &mut Pen) -> Result<(), FontError> {
    let args = take_width(pen, true)?;
    let [dx] = one(args)?;
    pen.move_to(dx, 0.0)
}

fn vmoveto(pen: &mut Pen) -> Result<(), FontError> {
    let args = take_width(pen, true)?;
    let [dy] = one(args)?;
    pen.move_to(0.0, dy)
}

fn rmoveto(pen: &mut Pen) -> Result<(), FontError> {
    let args = take_width(pen, false)?;
    let [dx, dy] = two(args)?;
    pen.move_to(dx, dy)
}

fn endchar(pen: &mut Pen) -> Result<(), FontError> {
    let args = take_width(pen, false)?;
    if !args.is_empty() {
        return Err(FontError::Unsupported);
    }
    pen.close();
    Ok(())
}

fn rlineto(pen: &mut Pen) -> Result<(), FontError> {
    let args = path_args(pen)?;
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err(FontError::BadTable);
    }
    for pair in args.chunks(2) {
        pen.line(pair[0], pair[1])?;
    }
    Ok(())
}

fn alternate_lines(pen: &mut Pen, mut horizontal: bool) -> Result<(), FontError> {
    let args = path_args(pen)?;
    if args.is_empty() {
        return Err(FontError::BadTable);
    }
    for delta in args {
        if horizontal {
            pen.line(delta, 0.0)?;
        } else {
            pen.line(0.0, delta)?;
        }
        horizontal = !horizontal;
    }
    Ok(())
}

fn rrcurveto(pen: &mut Pen) -> Result<(), FontError> {
    let args = path_args(pen)?;
    if args.is_empty() || !args.len().is_multiple_of(6) {
        return Err(FontError::BadTable);
    }
    for curve in args.chunks(6) {
        pen.curve(curve[0], curve[1], curve[2], curve[3], curve[4], curve[5])?;
    }
    Ok(())
}

fn rcurveline(pen: &mut Pen) -> Result<(), FontError> {
    let args = path_args(pen)?;
    if args.len() < 8 || !(args.len() - 2).is_multiple_of(6) {
        return Err(FontError::BadTable);
    }
    let (curves, line) = args.split_at(args.len() - 2);
    for curve in curves.chunks(6) {
        pen.curve(curve[0], curve[1], curve[2], curve[3], curve[4], curve[5])?;
    }
    pen.line(line[0], line[1])
}

fn rlinecurve(pen: &mut Pen) -> Result<(), FontError> {
    let args = path_args(pen)?;
    if args.len() < 6 || !(args.len() - 6).is_multiple_of(2) {
        return Err(FontError::BadTable);
    }
    let (lines, curve) = args.split_at(args.len() - 6);
    for line in lines.chunks(2) {
        pen.line(line[0], line[1])?;
    }
    pen.curve(curve[0], curve[1], curve[2], curve[3], curve[4], curve[5])
}

fn hhcurveto(pen: &mut Pen) -> Result<(), FontError> {
    let mut args = path_args(pen)?;
    let mut dy = 0.0;
    if args.len() % 4 == 1 {
        dy = args.remove(0);
    }
    if args.is_empty() || !args.len().is_multiple_of(4) {
        return Err(FontError::BadTable);
    }
    for curve in args.chunks(4) {
        pen.curve(curve[0], dy, curve[1], curve[2], curve[3], 0.0)?;
        dy = 0.0;
    }
    Ok(())
}

fn vvcurveto(pen: &mut Pen) -> Result<(), FontError> {
    let mut args = path_args(pen)?;
    let mut dx = 0.0;
    if args.len() % 4 == 1 {
        dx = args.remove(0);
    }
    if args.is_empty() || !args.len().is_multiple_of(4) {
        return Err(FontError::BadTable);
    }
    for curve in args.chunks(4) {
        pen.curve(dx, curve[0], curve[1], curve[2], 0.0, curve[3])?;
        dx = 0.0;
    }
    Ok(())
}

fn hvcurveto(pen: &mut Pen, mut horizontal: bool) -> Result<(), FontError> {
    let mut args = path_args(pen)?;
    if args.len() < 4 {
        return Err(FontError::BadTable);
    }
    while !args.is_empty() {
        if args.len() < 4 {
            return Err(FontError::BadTable);
        }
        let a = args.remove(0);
        let b = args.remove(0);
        let c = args.remove(0);
        let d = args.remove(0);
        let extra = if args.len() == 1 {
            Some(args.remove(0))
        } else {
            None
        };
        if horizontal {
            pen.curve(a, 0.0, b, c, extra.unwrap_or(0.0), d)?;
        } else {
            pen.curve(0.0, a, b, c, d, extra.unwrap_or(0.0))?;
        }
        horizontal = !horizontal;
    }
    Ok(())
}

fn escape(pen: &mut Pen, operator: u8) -> Result<(), FontError> {
    match operator {
        0 => Ok(()),
        3 => logic(pen, Logic::And),
        4 => logic(pen, Logic::Or),
        5 => logic(pen, Logic::Not),
        9 => unary(pen, Unary::Abs),
        10 => binary(pen, Binary::Add),
        11 => binary(pen, Binary::Sub),
        12 => binary(pen, Binary::Div),
        14 => unary(pen, Unary::Neg),
        15 => binary(pen, Binary::Eq),
        18 => {
            pen.pop()?;
            Ok(())
        }
        20 => put(pen),
        21 => get(pen),
        22 => ifelse(pen),
        23 => Err(FontError::Unsupported),
        24 => binary(pen, Binary::Mul),
        26 => unary(pen, Unary::Sqrt),
        27 => {
            let value = pen.pop()?;
            pen.push(value)?;
            pen.push(value)
        }
        28 => exch(pen),
        29 => index(pen),
        30 => roll(pen),
        34 => hflex(pen),
        35 => flex(pen, 13),
        36 => hflex1(pen),
        37 => flex1(pen),
        _ => Err(FontError::Unsupported),
    }
}

enum Logic {
    And,
    Or,
    Not,
}

enum Unary {
    Abs,
    Neg,
    Sqrt,
}

enum Binary {
    Add,
    Sub,
    Div,
    Mul,
    Eq,
}

fn logic(pen: &mut Pen, kind: Logic) -> Result<(), FontError> {
    let value = match kind {
        Logic::Not => {
            let number = pen.pop()?;
            if number == 0.0 { 1.0 } else { 0.0 }
        }
        Logic::And => {
            let right = pen.pop()?;
            let left = pen.pop()?;
            if left != 0.0 && right != 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Logic::Or => {
            let right = pen.pop()?;
            let left = pen.pop()?;
            if left != 0.0 || right != 0.0 {
                1.0
            } else {
                0.0
            }
        }
    };
    pen.push(value)
}

fn unary(pen: &mut Pen, kind: Unary) -> Result<(), FontError> {
    let number = pen.pop()?;
    let value = match kind {
        Unary::Abs => number.abs(),
        Unary::Neg => -number,
        Unary::Sqrt => {
            if number < 0.0 {
                return Err(FontError::BadTable);
            }
            number.sqrt()
        }
    };
    pen.push(value)
}

fn binary(pen: &mut Pen, kind: Binary) -> Result<(), FontError> {
    let right = pen.pop()?;
    let left = pen.pop()?;
    let value = match kind {
        Binary::Add => left + right,
        Binary::Sub => left - right,
        Binary::Mul => left * right,
        Binary::Div => {
            if right == 0.0 {
                return Err(FontError::BadTable);
            }
            left / right
        }
        Binary::Eq => {
            if left == right {
                1.0
            } else {
                0.0
            }
        }
    };
    pen.push(finite(value)?)
}

fn put(pen: &mut Pen) -> Result<(), FontError> {
    let index = transient_index(pen.pop()?)?;
    let value = pen.pop()?;
    pen.transient[index] = value;
    Ok(())
}

fn get(pen: &mut Pen) -> Result<(), FontError> {
    let index = transient_index(pen.pop()?)?;
    pen.push(pen.transient[index])
}

fn transient_index(value: f32) -> Result<usize, FontError> {
    if !(0.0..TRANSIENT as f32).contains(&value) || value.fract() != 0.0 {
        return Err(FontError::BadTable);
    }
    Ok(usize::from(value as u8))
}

fn ifelse(pen: &mut Pen) -> Result<(), FontError> {
    let v2 = pen.pop()?;
    let v1 = pen.pop()?;
    let second = pen.pop()?;
    let first = pen.pop()?;
    pen.push(if v1 <= v2 { first } else { second })
}

fn exch(pen: &mut Pen) -> Result<(), FontError> {
    let top = pen.pop()?;
    let next = pen.pop()?;
    pen.push(top)?;
    pen.push(next)
}

fn index(pen: &mut Pen) -> Result<(), FontError> {
    let which = whole(pen.pop()?)?;
    if which < 0 {
        return Err(FontError::BadTable);
    }
    let which = usize::try_from(which).map_err(|_| FontError::BadTable)?;
    let Some(from) = pen.stack.len().checked_sub(which + 1) else {
        return Err(FontError::BadTable);
    };
    let value = *pen.stack.get(from).ok_or(FontError::BadTable)?;
    pen.push(value)
}

fn roll(pen: &mut Pen) -> Result<(), FontError> {
    let shift = whole(pen.pop()?)?;
    let count = whole(pen.pop()?)?;
    if count < 0 {
        return Err(FontError::BadTable);
    }
    let count = usize::try_from(count).map_err(|_| FontError::BadTable)?;
    if count == 0 || count > pen.stack.len() {
        return Err(FontError::BadTable);
    }
    let start = pen.stack.len() - count;
    let turns = shift.rem_euclid(i32::try_from(count).map_err(|_| FontError::BadTable)?);
    let turns = usize::try_from(turns).map_err(|_| FontError::BadTable)?;
    pen.stack[start..].rotate_right(turns);
    Ok(())
}

fn hflex(pen: &mut Pen) -> Result<(), FontError> {
    let args = exactly(path_args(pen)?, 7)?;
    pen.curve(args[0], 0.0, args[1], args[2], args[3], 0.0)?;
    pen.curve(args[4], 0.0, args[5], -args[2], args[6], 0.0)
}

fn hflex1(pen: &mut Pen) -> Result<(), FontError> {
    let args = exactly(path_args(pen)?, 9)?;
    let dy = -(args[1] + args[3] + args[7]);
    pen.curve(args[0], args[1], args[2], args[3], args[4], 0.0)?;
    pen.curve(args[5], 0.0, args[6], args[7], args[8], dy)
}

fn flex(pen: &mut Pen, expected: usize) -> Result<(), FontError> {
    let args = exactly(path_args(pen)?, expected)?;
    pen.curve(args[0], args[1], args[2], args[3], args[4], args[5])?;
    pen.curve(args[6], args[7], args[8], args[9], args[10], args[11])
}

fn flex1(pen: &mut Pen) -> Result<(), FontError> {
    let args = exactly(path_args(pen)?, 11)?;
    let dx = args[0] + args[2] + args[4] + args[6] + args[8];
    let dy = args[1] + args[3] + args[5] + args[7] + args[9];
    let (dx6, dy6) = if dx.abs() > dy.abs() {
        (args[10], -dy)
    } else {
        (-dx, args[10])
    };
    pen.curve(args[0], args[1], args[2], args[3], args[4], args[5])?;
    pen.curve(args[6], args[7], args[8], args[9], dx6, dy6)
}

fn call_subr(cff: &Cff<'_>, pen: &mut Pen, depth: u32, global: bool) -> Result<Stop, FontError> {
    let encoded = whole(pen.pop()?)?;
    let index = if global { &cff.gsubrs } else { &cff.subrs };
    let biased = encoded
        .checked_add(bias(index.count))
        .ok_or(FontError::BadTable)?;
    if biased < 0 {
        return Err(FontError::BadTable);
    }
    let program = index.get(usize::try_from(biased).map_err(|_| FontError::BadTable)?)?;
    interpret(program, cff, pen, depth + 1)
}

fn bias(count: usize) -> i32 {
    if count < 1_240 {
        107
    } else if count < 33_900 {
        1_131
    } else {
        32_768
    }
}

fn path_args(pen: &mut Pen) -> Result<Vec<f32>, FontError> {
    if !pen.width_seen {
        pen.width_seen = true;
    }
    Ok(std::mem::take(&mut pen.stack))
}

fn take_width(pen: &mut Pen, width_when_even: bool) -> Result<Vec<f32>, FontError> {
    let mut args = std::mem::take(&mut pen.stack);
    if !pen.width_seen {
        pen.width_seen = true;
        let even = args.len().is_multiple_of(2);
        if even == width_when_even {
            if args.is_empty() {
                return Err(FontError::BadTable);
            }
            args.remove(0);
        }
    }
    Ok(args)
}

fn one(args: Vec<f32>) -> Result<[f32; 1], FontError> {
    args.try_into().map_err(|_| FontError::BadTable)
}

fn two(args: Vec<f32>) -> Result<[f32; 2], FontError> {
    args.try_into().map_err(|_| FontError::BadTable)
}

fn exactly(args: Vec<f32>, count: usize) -> Result<Vec<f32>, FontError> {
    if args.len() == count {
        Ok(args)
    } else {
        Err(FontError::BadTable)
    }
}

fn whole(value: f32) -> Result<i32, FontError> {
    if !value.is_finite()
        || value.fract() != 0.0
        || !(f32::from(i16::MIN)..=f32::from(i16::MAX)).contains(&value)
    {
        return Err(FontError::BadTable);
    }
    Ok(value as i32)
}
