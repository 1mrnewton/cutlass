//! Inter field pictures against FFmpeg's decode: P and B field pictures
//! with field and 16×8 prediction, field dual-prime, a P field that
//! predicts from the first field of its own frame, skipped macroblocks in P
//! and B fields, and a P frame picture with frame, field, and dual-prime
//! prediction.
//!
//! `inter_fields.m2v` is written below, bit by bit, from H.262 (1995)
//! clauses 6.2.5 and 7.6, so it is ours to commit. It is a 64×64
//! interlaced Main profile sequence, five frames in one closed GOP, coded
//! I P P P B:
//!
//! - frame 0: an I field pair. Every 8×8 block has its own DC level, and
//!   the two fields differ, so a wrong vector, field, or average shows.
//! - frame 1: a P field pair with field, 16×8, and dual-prime prediction
//!   and a skipped macroblock. Its bottom field predicts from its own top
//!   field.
//! - frame 2: a P frame picture, `top_field_first`, with frame, field, and
//!   dual-prime prediction and a skipped macroblock.
//! - frame 4: a P field pair with field and 16×8 prediction and skips; no
//!   dual prime, because a B frame sits between it and its reference.
//! - frame 3: a B field pair with forward, backward, and bidirectional
//!   field and 16×8 prediction, and skipped macroblocks that repeat the
//!   macroblock before them.
//!
//! No inter macroblock carries a residual, so a decoder's output is the
//! prediction alone, and every vector keeps the prediction inside the
//! reference (7.6.3.8). `inter_fields.yuv` is FFmpeg 8.1.1's decode
//! (`ffmpeg -xerror -i inter_fields.m2v -f rawvideo -pix_fmt yuv420p`),
//! frames in
//! display order. Prediction is exact integer arithmetic and the intra
//! blocks are DC only, so the check is equality.
//!
//! With FFmpeg on the machine, the last test decodes the fixture again and
//! compares it with the committed reference. Without FFmpeg it says so and
//! skips, or fails when `MPEG2_REQUIRE_ORACLES` is set.

#[cfg(test)]
mod oracles;

#[cfg(test)]
mod inter_fields {
    use std::process::Command;

    use cutlass_decode::{Packet, VideoOutput};
    use cutlass_mpeg2::Mpeg2Decoder;
    use cutlass_mpeg2::es::split;

    const W: usize = 64;
    const H: usize = 64;
    /// Luma rows in one field.
    const FIELD_H: usize = H / 2;
    const FRAME_BYTES: usize = W * H * 3 / 2;

    const TOP: u8 = 1;
    const BOTTOM: u8 = 2;
    const FRAME: u8 = 3;

    const I: u32 = 1;
    const P: u32 = 2;
    const B: u32 = 3;

    #[derive(Default)]
    struct Bits {
        bytes: Vec<u8>,
        acc: u64,
        len: u32,
    }

    impl Bits {
        fn put(&mut self, value: u32, bits: u32) {
            self.acc = (self.acc << bits) | u64::from(value);
            self.len += bits;
            while self.len >= 8 {
                self.len -= 8;
                self.bytes.push((self.acc >> self.len) as u8);
            }
        }

        fn code(&mut self, pattern: &str) {
            for bit in pattern.bytes().filter(|&b| b != b' ') {
                self.put(u32::from(bit == b'1'), 1);
            }
        }

        fn start(&mut self, code: u8) {
            self.align();
            self.bytes.extend_from_slice(&[0, 0, 1, code]);
        }

        fn align(&mut self) {
            if self.len > 0 {
                self.put(0, 8 - self.len);
            }
        }
    }

    /// Table B.1, macroblock_address_increment 1 to 4.
    fn address_increment(increment: usize) -> &'static str {
        ["1", "011", "010", "0011"][increment - 1]
    }

    /// Table B.10, motion_code -16 to 16.
    fn motion_code(code: i32) -> &'static str {
        const CODES: [&str; 33] = [
            "0000 0011 001",
            "0000 0011 011",
            "0000 0011 101",
            "0000 0011 111",
            "0000 0100 001",
            "0000 0100 011",
            "0000 0100 11",
            "0000 0101 01",
            "0000 0101 11",
            "0000 0111",
            "0000 1001",
            "0000 1011",
            "0000 111",
            "0001 1",
            "0011",
            "011",
            "1",
            "010",
            "0010",
            "0001 0",
            "0000 110",
            "0000 1010",
            "0000 1000",
            "0000 0110",
            "0000 0101 10",
            "0000 0101 00",
            "0000 0100 10",
            "0000 0100 010",
            "0000 0100 000",
            "0000 0011 110",
            "0000 0011 100",
            "0000 0011 010",
            "0000 0011 000",
        ];
        CODES[(code + 16) as usize]
    }

    /// Table B.11, dmvector.
    fn dmvector(value: i32) -> &'static str {
        match value {
            0 => "0",
            1 => "10",
            -1 => "11",
            _ => panic!("dmvector {value}"),
        }
    }

    /// Tables B.12 and B.13, dct_dc_size for luma and chroma.
    fn dc_size(size: u32, chroma: bool) -> &'static str {
        let luma = [
            "100", "00", "01", "101", "110", "1110", "11110", "111110", "1111110",
        ];
        let chroma_codes = [
            "00", "01", "10", "110", "1110", "11110", "111110", "1111110", "11111110",
        ];
        if chroma {
            chroma_codes[size as usize]
        } else {
            luma[size as usize]
        }
    }

    /// A vector in half samples: horizontal, vertical.
    type Vector = (i32, i32);
    /// A reference field (0 top, 1 bottom) and the vector into it.
    type Pick = (u32, Vector);

    #[derive(Clone, Copy)]
    enum Motion {
        /// Field picture: the whole field from one reference field.
        Field(Pick),
        /// Field picture: the upper and lower 16×8 halves.
        Halves([Pick; 2]),
        /// Frame picture: the top field, then the bottom.
        Fields([Pick; 2]),
        /// Frame picture: frame prediction.
        Frame(Vector),
        /// Dual prime: the same-parity field vector and dmvector.
        Dual(Vector, Vector),
    }

    #[derive(Clone, Copy)]
    enum Mb {
        /// Nothing coded: the decoder forms it from the rules in 7.6.6.
        Skip,
        /// MC, Not Coded, in a P picture.
        P(Motion),
        /// Not Coded, in a B picture: forward, backward, or both, with the
        /// same motion type.
        B(Option<Motion>, Option<Motion>),
    }

    /// The intra level of an 8×8 block: distinct per block, plane, and
    /// field, inside 32..=223 so no prediction saturates.
    fn level(parity: usize, plane: usize, bx: usize, by: usize) -> i32 {
        let seed = (parity * 7919 + plane * 104_729 + by * 613 + bx * 37) as u32;
        let mixed = seed.wrapping_mul(2_654_435_761) >> 24;
        32 + (mixed % 192) as i32
    }

    /// A picture's position in the stream and what its references are,
    /// for the bounds check.
    struct Picture {
        temporal_reference: u32,
        kind: u32,
        structure: u8,
    }

    fn sequence() -> Vec<u8> {
        let mut w = Bits::default();
        w.start(0xb3);
        w.put(W as u32, 12);
        w.put(H as u32, 12);
        w.put(1, 4); // square samples
        w.put(3, 4); // 25 Hz
        w.put(1000, 18);
        w.put(1, 1);
        w.put(10, 10);
        w.put(0, 3); // not constrained, default matrices
        w.start(0xb5);
        w.put(1, 4);
        w.put(0x48, 8); // Main profile, Main level
        w.put(0, 1); // progressive_sequence 0: interlaced
        w.put(1, 2); // 4:2:0
        w.put(0, 4);
        w.put(0, 12);
        w.put(1, 1);
        w.put(0, 8);
        w.put(0, 1); // low_delay
        w.put(0, 7);
        w.start(0xb8);
        w.put(0, 12); // drop_frame_flag, hours, minutes
        w.put(1, 1); // marker
        w.put(0, 12); // seconds, pictures
        w.put(1, 1); // closed_gop
        w.put(0, 1);
        w.align();
        w.bytes
    }

    fn picture_header(w: &mut Bits, picture: &Picture) {
        w.start(0x00);
        w.put(picture.temporal_reference, 10);
        w.put(picture.kind, 3);
        w.put(0xffff, 16); // vbv_delay
        if picture.kind != I {
            w.put(0, 1); // full_pel_forward_vector
            w.put(7, 3); // forward_f_code, unused in MPEG-2
        }
        if picture.kind == B {
            w.put(0, 1);
            w.put(7, 3);
        }
        w.put(0, 1); // extra_bit_picture
        w.start(0xb5);
        w.put(8, 4);
        let forward = if picture.kind == I { 15 } else { 1 };
        let backward = if picture.kind == B { 1 } else { 15 };
        for f_code in [forward, forward, backward, backward] {
            w.put(f_code, 4);
        }
        w.put(0, 2); // 8-bit DC
        w.put(u32::from(picture.structure), 2);
        let frame = picture.structure == FRAME;
        w.put(u32::from(frame), 1); // top_field_first
        w.put(0, 1); // frame_pred_frame_dct
        w.put(0, 6); // concealment, q_scale_type, intra VLC, scan, repeat, 420 type
        w.put(0, 1); // progressive_frame
        w.put(0, 1); // composite_display_flag
    }

    /// Writes one DC difference, with its predictor kept in `pred`.
    fn dc(w: &mut Bits, pred: &mut i32, value: i32, chroma: bool) {
        let diff = value - *pred;
        *pred = value;
        let size = 32 - diff.unsigned_abs().leading_zeros();
        w.code(dc_size(size, chroma));
        if size > 0 {
            let bits = if diff > 0 {
                diff
            } else {
                diff + (1 << size) - 1
            };
            w.put(bits as u32, size);
        }
        w.code("10"); // end of block
    }

    /// An I field: two slices of four intra macroblocks, DC only.
    fn intra_field(parity: usize, temporal_reference: u32) -> Vec<u8> {
        let mut w = Bits::default();
        let structure = if parity == 0 { TOP } else { BOTTOM };
        picture_header(
            &mut w,
            &Picture {
                temporal_reference,
                kind: I,
                structure,
            },
        );
        for row in 0..FIELD_H / 16 {
            w.start(row as u8 + 1);
            w.put(8, 5);
            w.put(0, 1);
            let mut pred = [128; 3];
            for col in 0..W / 16 {
                w.code(address_increment(1));
                w.code("1"); // intra
                for block in 0..4 {
                    let (bx, by) = (col * 2 + block % 2, row * 2 + block / 2);
                    dc(&mut w, &mut pred[0], level(parity, 0, bx, by), false);
                }
                for (plane, pred) in pred.iter_mut().enumerate().skip(1) {
                    dc(&mut w, pred, level(parity, plane, col, row), true);
                }
            }
        }
        w.align();
        w.bytes
    }

    /// Integer division rounding half away from zero (`//` in H.262).
    fn round_half(value: i32) -> i32 {
        if value >= 0 {
            (value + 1) / 2
        } else {
            -((-value + 1) / 2)
        }
    }

    /// Checks that a `width` × `height` luma block at (`x`, `y`) moved by
    /// `vector`, and its chroma, read only samples of a `plane_w` ×
    /// `plane_h` reference.
    fn inside(at: (usize, usize), size: (usize, usize), vector: Vector, plane: (usize, usize)) {
        let check = |x: i32, y: i32, w: i32, h: i32, v: Vector, pw: i32, ph: i32| {
            let left = x + v.0.div_euclid(2);
            let top = y + v.1.div_euclid(2);
            let right = left + w - 1 + i32::from(v.0 % 2 != 0);
            let bottom = top + h - 1 + i32::from(v.1 % 2 != 0);
            assert!(
                left >= 0 && top >= 0 && right < pw && bottom < ph,
                "{w}x{h} at ({x}, {y}) by {v:?} leaves {pw}x{ph}"
            );
        };
        let (x, y) = (at.0 as i32, at.1 as i32);
        let (w, h) = (size.0 as i32, size.1 as i32);
        let (pw, ph) = (plane.0 as i32, plane.1 as i32);
        check(x, y, w, h, vector, pw, ph);
        let chroma = (vector.0 / 2, vector.1 / 2);
        check(x / 2, y / 2, w / 2, h / 2, chroma, pw / 2, ph / 2);
    }

    /// Writes one component and updates its predictor (7.6.3.1, f_code 1).
    /// `halve` is a field vector's vertical part in a frame picture.
    fn component(w: &mut Bits, pmv: &mut i32, value: i32, halve: bool) {
        let prediction = if halve { pmv.div_euclid(2) } else { *pmv };
        let mut delta = value - prediction;
        if delta < -16 {
            delta += 32;
        }
        if delta > 15 {
            delta -= 32;
        }
        // A field vector's vertical part in a frame picture covers half
        // the range (7.6.3.2).
        let range = if halve { -8..=7 } else { -16..=15 };
        assert!(range.contains(&value), "vector {value}");
        w.code(motion_code(delta));
        *pmv = if halve { value * 2 } else { value };
    }

    /// The vector for `r`, `s` with its predictors `pmv[r][s]`.
    fn vector(
        w: &mut Bits,
        pmv: &mut [[Vector; 2]; 2],
        r: usize,
        s: usize,
        v: Vector,
        frame_field: bool,
    ) {
        let (mut h, mut vert) = pmv[r][s];
        component(w, &mut h, v.0, false);
        component(w, &mut vert, v.1, frame_field);
        pmv[r][s] = (h, vert);
    }

    /// The dual-prime vector of opposite parity (7.6.3.6).
    fn opposite(v: Vector, dmv: Vector, m: i32, e: i32) -> Vector {
        (round_half(v.0 * m) + dmv.0, round_half(v.1 * m) + e + dmv.1)
    }

    /// Writes the motion type, vectors, and predictor updates of one
    /// direction `s`, and checks the reads stay in the references.
    fn motion(
        w: &mut Bits,
        pmv: &mut [[Vector; 2]; 2],
        s: usize,
        motion: Motion,
        frame: bool,
        at: (usize, usize),
    ) {
        let field_plane = (W, FIELD_H);
        match motion {
            Motion::Field((select, v)) => {
                assert!(!frame);
                w.put(select, 1);
                vector(w, pmv, 0, s, v, false);
                pmv[1][s] = pmv[0][s];
                inside(at, (16, 16), v, field_plane);
            }
            Motion::Halves(halves) => {
                assert!(!frame);
                for (r, (select, v)) in halves.into_iter().enumerate() {
                    w.put(select, 1);
                    vector(w, pmv, r, s, v, false);
                    inside((at.0, at.1 + 8 * r), (16, 8), v, field_plane);
                }
            }
            Motion::Fields(fields) => {
                assert!(frame);
                for (r, (select, v)) in fields.into_iter().enumerate() {
                    w.put(select, 1);
                    vector(w, pmv, r, s, v, true);
                    inside((at.0, at.1 / 2), (16, 8), v, field_plane);
                }
            }
            Motion::Frame(v) => {
                assert!(frame);
                vector(w, pmv, 0, s, v, false);
                pmv[1][s] = pmv[0][s];
                inside(at, (16, 16), v, (W, H));
            }
            Motion::Dual(v, dmv) => {
                assert_eq!(s, 0, "dual prime is forward only");
                let (mut h, mut vert) = pmv[0][0];
                component(w, &mut h, v.0, false);
                w.code(dmvector(dmv.0));
                component(w, &mut vert, v.1, frame);
                w.code(dmvector(dmv.1));
                pmv[0][0] = (h, vert);
                pmv[1][0] = pmv[0][0];
                if frame {
                    // top_field_first: the top field's opposite parity is
                    // one field back (m 1, e -1), the bottom's three (m 3,
                    // e +1).
                    let block = (at.0, at.1 / 2);
                    inside(block, (16, 8), v, field_plane);
                    inside(block, (16, 8), opposite(v, dmv, 1, -1), field_plane);
                    inside(block, (16, 8), opposite(v, dmv, 3, 1), field_plane);
                } else {
                    inside(at, (16, 16), v, field_plane);
                    inside(at, (16, 16), opposite(v, dmv, 1, 1), field_plane);
                    inside(at, (16, 16), opposite(v, dmv, 1, -1), field_plane);
                }
            }
        }
    }

    /// The two-bit frame_motion_type or field_motion_type.
    fn motion_type(motion: Motion) -> u32 {
        match motion {
            Motion::Field(_) | Motion::Fields(_) => 1,
            Motion::Halves(_) | Motion::Frame(_) => 2,
            Motion::Dual(..) => 3,
        }
    }

    /// A P or B picture whose macroblocks are `rows`, one slice a row.
    fn inter_picture(picture: &Picture, rows: &[[Mb; 4]]) -> Vec<u8> {
        let frame = picture.structure == FRAME;
        let mb_rows = if frame { H / 16 } else { FIELD_H / 16 };
        assert_eq!(rows.len(), mb_rows);
        let mut w = Bits::default();
        picture_header(&mut w, picture);
        for (row, mbs) in rows.iter().enumerate() {
            w.start(row as u8 + 1);
            w.put(8, 5);
            w.put(0, 1);
            let mut pmv = [[(0, 0); 2]; 2];
            let mut increment = 1;
            for (col, mb) in mbs.iter().enumerate() {
                let at = (col * 16, row * 16);
                match *mb {
                    Mb::Skip => {
                        assert!(col > 0 && col < 3, "a slice starts and ends coded");
                        if picture.kind == P {
                            pmv = [[(0, 0); 2]; 2];
                        }
                        increment += 1;
                        continue;
                    }
                    Mb::P(m) => {
                        assert_eq!(picture.kind, P);
                        w.code(address_increment(increment));
                        w.code("001"); // MC, Not Coded
                        w.put(motion_type(m), 2);
                        motion(&mut w, &mut pmv, 0, m, frame, at);
                    }
                    Mb::B(forward, backward) => {
                        assert_eq!(picture.kind, B);
                        w.code(address_increment(increment));
                        w.code(match (forward, backward) {
                            (Some(_), Some(_)) => "10",
                            (None, Some(_)) => "010",
                            (Some(_), None) => "0010",
                            (None, None) => panic!("a B macroblock predicts"),
                        });
                        let kind = forward.or(backward).map(motion_type);
                        if let (Some(f), Some(b)) = (forward, backward) {
                            assert_eq!(motion_type(f), motion_type(b));
                        }
                        w.put(kind.unwrap_or(1), 2);
                        for (s, m) in [forward, backward].into_iter().enumerate() {
                            if let Some(m) = m {
                                motion(&mut w, &mut pmv, s, m, frame, at);
                            }
                        }
                    }
                }
                increment = 1;
            }
        }
        w.align();
        w.bytes
    }

    fn field(select: u32, v: Vector) -> Motion {
        Motion::Field((select, v))
    }

    fn halves(upper: Pick, lower: Pick) -> Motion {
        Motion::Halves([upper, lower])
    }

    fn fields(top: Pick, bottom: Pick) -> Motion {
        Motion::Fields([top, bottom])
    }

    fn dual(v: Vector, dmv: Vector) -> Motion {
        Motion::Dual(v, dmv)
    }

    fn p(m: Motion) -> Mb {
        Mb::P(m)
    }

    fn bi(forward: Motion, backward: Motion) -> Mb {
        Mb::B(Some(forward), Some(backward))
    }

    fn fwd(m: Motion) -> Mb {
        Mb::B(Some(m), None)
    }

    fn bwd(m: Motion) -> Mb {
        Mb::B(None, Some(m))
    }

    const SKIP: Mb = Mb::Skip;

    fn picture(temporal_reference: u32, kind: u32, structure: u8) -> Picture {
        Picture {
            temporal_reference,
            kind,
            structure,
        }
    }

    /// The whole stream, written from H.262.
    fn stream() -> Vec<u8> {
        let mut out = sequence();
        // Frame 0: the I pair.
        out.extend(intra_field(0, 0));
        out.extend(intra_field(1, 0));
        // Frame 1: P fields. The top field reads frame 0; the bottom reads
        // this frame's top field (select 0) and frame 0's bottom.
        out.extend(inter_picture(
            &picture(1, P, TOP),
            &[
                [
                    p(field(1, (5, 4))),
                    p(halves((0, (-3, 2)), (1, (4, -3)))),
                    p(dual((-4, 6), (1, -1))),
                    p(field(0, (-6, 3))),
                ],
                [
                    p(field(0, (2, -5))),
                    SKIP,
                    p(dual((3, -6), (0, 1))),
                    p(halves((1, (-1, -4)), (0, (-2, -7)))),
                ],
            ],
        ));
        out.extend(inter_picture(
            &picture(1, P, BOTTOM),
            &[
                [
                    p(halves((0, (4, 5)), (1, (1, -2)))),
                    p(dual((-2, 8), (-1, 0))),
                    p(field(0, (7, 1))),
                    p(field(1, (-5, 2))),
                ],
                [
                    p(dual((6, -8), (1, 1))),
                    p(field(1, (0, -1))),
                    SKIP,
                    p(field(0, (-8, -8))),
                ],
            ],
        ));
        // Frame 2: a P frame picture from frame 1.
        out.extend(inter_picture(
            &picture(2, P, FRAME),
            &[
                [
                    p(Motion::Frame((3, 4))),
                    p(fields((1, (2, 3)), (0, (-2, 5)))),
                    p(dual((-3, 4), (0, -1))),
                    p(Motion::Frame((-4, 0))),
                ],
                [
                    p(dual((4, -2), (1, 0))),
                    SKIP,
                    p(Motion::Frame((1, -3))),
                    p(fields((0, (-1, -2)), (1, (-3, 7)))),
                ],
                [
                    p(fields((1, (0, 1)), (1, (5, -4)))),
                    p(dual((-2, -3), (-1, 1))),
                    p(Motion::Frame((-5, 6))),
                    p(dual((-6, 2), (1, 0))),
                ],
                [
                    p(dual((2, -4), (0, 0))),
                    p(Motion::Frame((2, -6))),
                    p(fields((0, (3, -8)), (1, (-4, -1)))),
                    p(Motion::Frame((-2, -2))),
                ],
            ],
        ));
        // Frame 4: P fields from frame 2, then from its own top field.
        out.extend(inter_picture(
            &picture(4, P, TOP),
            &[
                [
                    p(field(1, (4, 2))),
                    p(halves((0, (-2, 4)), (1, (6, -1)))),
                    SKIP,
                    p(halves((1, (-3, 0)), (0, (-1, -5)))),
                ],
                [
                    p(halves((1, (1, 1)), (0, (3, -6)))),
                    p(field(0, (-1, -3))),
                    p(field(1, (2, -2))),
                    p(field(0, (-4, -4))),
                ],
            ],
        ));
        out.extend(inter_picture(
            &picture(4, P, BOTTOM),
            &[
                [
                    p(field(0, (2, 3))),
                    p(halves((1, (5, 1)), (0, (-1, -4)))),
                    p(field(1, (3, 0))),
                    p(field(0, (-3, 6))),
                ],
                [
                    p(field(1, (6, -7))),
                    SKIP,
                    p(halves((0, (0, -2)), (1, (-5, -6)))),
                    p(field(1, (-2, -1))),
                ],
            ],
        ));
        // Frame 3: B fields, forward from frame 2, backward from frame 4.
        // A skipped macroblock repeats the direction before it, from the
        // same-parity field, so the macroblock before each skip picks the
        // same parity.
        out.extend(inter_picture(
            &picture(3, B, TOP),
            &[
                [
                    bi(field(0, (2, 2)), field(0, (4, 4))),
                    SKIP,
                    bwd(halves((1, (-2, 6)), (0, (3, -3)))),
                    bi(
                        halves((0, (-4, 1)), (1, (-6, -2))),
                        halves((1, (-1, 3)), (0, (-2, -1))),
                    ),
                ],
                [
                    bi(
                        halves((0, (1, -1)), (1, (2, -5))),
                        halves((1, (3, -2)), (0, (0, -7))),
                    ),
                    fwd(field(0, (3, -4))),
                    SKIP,
                    bwd(field(1, (-2, -6))),
                ],
            ],
        ));
        out.extend(inter_picture(
            &picture(3, B, BOTTOM),
            &[
                [
                    bi(
                        halves((1, (2, 3)), (0, (1, -2))),
                        halves((0, (5, 1)), (1, (1, -4))),
                    ),
                    bi(field(1, (-3, 5)), field(1, (2, 1))),
                    SKIP,
                    fwd(field(0, (-5, 2))),
                ],
                [
                    bwd(field(0, (4, -3))),
                    bi(
                        halves((1, (1, -2)), (0, (2, -3))),
                        halves((1, (-2, -1)), (0, (3, -6))),
                    ),
                    fwd(halves((0, (0, -4)), (1, (-4, -1)))),
                    bi(field(1, (-6, -5)), field(0, (-1, -1))),
                ],
            ],
        ));
        out.extend_from_slice(&[0, 0, 1, 0xb7]);
        out
    }

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    }

    fn decode(stream: &[u8]) -> Vec<VideoOutput> {
        let mut decoder = Mpeg2Decoder::default();
        let mut frames = Vec::new();
        for unit in split(stream).unwrap() {
            let packet = Packet {
                stream: 0,
                data: unit.data,
                pts: i64::try_from(unit.display_index).unwrap(),
                duration: 1,
                timescale: 25,
                keyframe: unit.coding_type == 1,
                side: Vec::new(),
            };
            frames.extend(decoder.push(&packet).unwrap());
        }
        frames.extend(decoder.drain().unwrap());
        frames
    }

    #[test]
    fn the_committed_stream_is_the_one_written_here() {
        let written = stream();
        let committed = std::fs::read(format!(
            "{}/tests/fixtures/inter_fields.m2v",
            env!("CARGO_MANIFEST_DIR")
        ))
        .ok();
        if committed.as_ref() != Some(&written) {
            let path = std::env::temp_dir().join("inter_fields.m2v");
            std::fs::write(&path, &written).unwrap();
            panic!("the stream changed; the new one is {}", path.display());
        }
    }

    #[test]
    fn inter_field_pictures_match_the_reference_decoder() {
        let reference = fixture("inter_fields.yuv");
        let frames = decode(&fixture("inter_fields.m2v"));
        assert_eq!(frames.len(), 5);
        assert_eq!(reference.len(), 5 * FRAME_BYTES);
        for (index, (frame, expected)) in
            frames.iter().zip(reference.chunks(FRAME_BYTES)).enumerate()
        {
            assert_eq!(frame.pts, index as i64, "display order");
            assert_eq!(
                (frame.picture.width(), frame.picture.height()),
                (W as u32, H as u32)
            );
            let planes = [frame.picture.y(), frame.picture.u(), frame.picture.v()];
            let mut offset = 0;
            for (p, plane) in planes.into_iter().enumerate() {
                let want = &expected[offset..offset + plane.len()];
                offset += plane.len();
                let first = plane.iter().zip(want).position(|(a, b)| a != b);
                assert!(
                    first.is_none(),
                    "frame {index} plane {p} sample {first:?}: {} against {}",
                    first.map_or(0, |i| plane[i]),
                    first.map_or(0, |i| want[i]),
                );
            }
        }
    }

    #[test]
    fn the_reference_frames_differ_so_each_tool_shows() {
        // Each inter frame moves its texture, so a decoder that copied the
        // previous frame, or one field into both, would not match.
        let reference = fixture("inter_fields.yuv");
        let frames: Vec<&[u8]> = reference.chunks(FRAME_BYTES).collect();
        for pair in frames.windows(2) {
            assert_ne!(pair[0], pair[1]);
        }
        for frame in &frames {
            let luma = &frame[..W * H];
            let even: Vec<u8> = luma.chunks(W).step_by(2).flatten().copied().collect();
            let odd: Vec<u8> = luma
                .chunks(W)
                .skip(1)
                .step_by(2)
                .flatten()
                .copied()
                .collect();
            assert_ne!(even, odd);
        }
    }

    #[test]
    fn ffmpeg_decodes_the_stream_to_the_committed_reference() {
        let Some(ffmpeg) = crate::oracles::ffmpeg() else {
            return;
        };
        let dir =
            std::env::temp_dir().join(format!("cutlass-mpeg2-inter-fields-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let yuv = dir.join("ffmpeg.yuv");
        let out = Command::new(&ffmpeg)
            .args(["-y", "-hide_banner", "-loglevel", "error"])
            // `-err_detect explode` refuses every field picture, FATE's
            // own `t.mpg` among them, so only `-xerror` is set.
            .args(["-xerror", "-i"])
            .arg(format!(
                "{}/tests/fixtures/inter_fields.m2v",
                env!("CARGO_MANIFEST_DIR")
            ))
            .args(["-fps_mode", "passthrough", "-f", "rawvideo"])
            .args(["-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let theirs = std::fs::read(&yuv).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            theirs == fixture("inter_fields.yuv"),
            "FFmpeg's decode changed"
        );

        // ffprobe reads the hand-written headers the same way: five
        // interlaced frames, I P P B P in display order.
        let Some(ffprobe) = crate::oracles::ffprobe(&ffmpeg) else {
            return;
        };
        let probe = Command::new(ffprobe)
            .args(["-v", "error", "-show_entries"])
            .args(["frame=pict_type,interlaced_frame", "-of", "csv=p=0"])
            .arg(format!(
                "{}/tests/fixtures/inter_fields.m2v",
                env!("CARGO_MANIFEST_DIR")
            ))
            .output()
            .unwrap();
        assert!(
            probe.status.success(),
            "{}",
            String::from_utf8_lossy(&probe.stderr)
        );
        let text = String::from_utf8_lossy(&probe.stdout);
        let frames: Vec<&str> = text
            .lines()
            .map(|line| line.trim_end_matches(','))
            .collect();
        assert_eq!(frames, ["I,1", "P,1", "P,1", "B,1", "P,1"], "{text}");
    }
}
