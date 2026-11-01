//! Annex G slice extension.
//!
//! A slice with no inter-layer prediction is the Annex A reconstruction.
//! Same-size intra base mode copies the reference layer and the enhancement
//! crop is the picture. FFmpeg 8.1.1 does not decode NAL type 20, so that
//! picture follows the recommendation. FFmpeg still emits the Annex A base.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder, VideoOutput};

use crate::H264Decoder;
use crate::nal::{annex_b_extension, annex_b_nal};
use crate::set::parse_subset_sps;

#[test]
fn a_slice_extension_without_inter_layer_prediction_is_a_picture() {
    for profile in [83u8, 86] {
        for restriction in [true, false] {
            let rbsp = subset(profile, restriction, 0, false);
            let parsed = parse_subset_sps(&rbsp).expect("subset");
            assert_eq!(parsed.slice_header_restriction, restriction);
            let annex = svc_annex(&rbsp, 0, true, restriction);
            let picture = decode(&annex);
            assert!(
                flat(&picture),
                "profile {profile} restriction {restriction}"
            );
            if let Some(ffmpeg) = ffmpeg_bin() {
                assert!(
                    ffmpeg_bytes(&ffmpeg, &annex).is_empty(),
                    "ffmpeg decoded a slice extension"
                );
            }
        }
    }
    let annex = svc_annex(&subset(83, true, 0, false), 1, true, true);
    assert!(flat(&decode(&annex)));
}

#[test]
fn output_flag_keeps_the_extension_off_the_display() {
    let annex = svc_annex(&subset(83, true, 0, false), 0, false, true);
    let mut decoder = open();
    let pictures = decoder
        .push(&Packet::video(annex, 0, 1, 30, true))
        .expect("decode");
    assert!(pictures.is_empty());
    assert!(decoder.drain().expect("drain").is_empty());
    let (y, u, v) = decoder.picture_at(0).expect("held");
    assert!(
        y.iter()
            .chain(u.iter())
            .chain(v.iter())
            .all(|sample| *sample == 128)
    );
}

#[test]
fn inter_layer_intra_prediction_copies_the_base() {
    let parsed = parse_subset_sps(&layer_subset()).expect("subset");
    assert!(parsed.inter_layer_deblock);
    assert_eq!(parsed.svc_spatial, 0);
    assert!(!parsed.seq_tcoeff);
    assert_eq!(parsed.visible().expect("visible"), (8, 8));
    let base = pcm_annex();
    let base_picture = decode(&base);
    assert_eq!(base_picture.picture.width(), 16);
    assert_eq!(base_picture.picture.height(), 16);
    assert!(painted(&base_picture, 200, 40));
    let annex = inter_layer_annex();
    let picture = decode(&annex);
    assert_eq!(picture.picture.width(), 8);
    assert_eq!(picture.picture.height(), 8);
    assert!(painted(&picture, 200, 40));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let raw = ffmpeg_raw(&ffmpeg, &base);
        assert_eq!(raw, paint(16, 200, 40));
        let layered = ffmpeg_raw(&ffmpeg, &annex);
        assert_eq!(layered, paint(16, 200, 40));
    }
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &annex);
    assert!(flat(&decode_with(&mut decoder, &base_idr())));
}

#[test]
fn intra_base_adds_the_coded_residual() {
    let annex = residual_annex();
    let picture = decode(&annex);
    assert_eq!(picture.picture.width(), 16);
    let y = picture.picture.y();
    for row in 0..16 {
        for col in 0..16 {
            let want = if row < 4 && col < 4 { 203 } else { 200 };
            assert_eq!(y[row * 16 + col], want, "{row},{col}");
        }
    }
    assert!(
        picture
            .picture
            .u()
            .iter()
            .chain(picture.picture.v())
            .all(|sample| *sample == 40)
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert_eq!(ffmpeg_raw(&ffmpeg, &annex), paint(16, 200, 40));
    }
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &annex);
    assert!(flat(&decode_with(&mut decoder, &base_idr())));
}

#[test]
fn inter_layer_motion_predicts_from_the_reference() {
    let idr = motion_idr();
    let mut decoder = open();
    let first = decode_with(&mut decoder, &idr);
    assert!(painted(&first, 150, 90));
    let base = decode_with(&mut decoder, &motion_base_packet());
    assert_eq!(base.picture.width(), 16);
    let base_y = base.picture.y();
    let corner = (0..4).all(|row| {
        base_y[row * 16..row * 16 + 4]
            .iter()
            .all(|sample| *sample == 153)
    });
    let elsewhere = (0..16).all(|row| {
        let start = if row < 4 { 4 } else { 0 };
        base_y[row * 16 + start..row * 16 + 16]
            .iter()
            .all(|sample| *sample == 150)
    });
    assert!(
        corner && elsewhere,
        "base residual did not land on the top-left 4×4"
    );
    assert!(base.picture.u().iter().all(|sample| *sample == 90));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &idr);
    let picture = decode_with(&mut decoder, &motion_layer_packet());
    assert!(painted(&picture, 150, 90));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut layered = idr.clone();
        layered.extend(motion_layer_packet());
        let raw = ffmpeg_raw(&ffmpeg, &layered);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame * 2, "ffmpeg picture count");
        let ffmpeg_base = &raw[frame..];
        assert_eq!(ffmpeg_base, &planes_of(&base));
        assert_ne!(ffmpeg_base, &planes_of(&picture));
    }
}

#[test]
fn inter_layer_residual_adds_the_base_difference() {
    let idr = residual_idr();
    let mut decoder = open();
    let first = decode_with(&mut decoder, &idr);
    assert!(split_luma(&first, 100, 180));
    let base = decode_with(&mut decoder, &motion_base_packet());
    assert!(base_residual_picture(&base));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &idr);
    let picture = decode_with(&mut decoder, &residual_layer_packet());
    assert!(shifted_residual_picture(&picture));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut layered = idr.clone();
        layered.extend(residual_layer_packet());
        let raw = ffmpeg_raw(&ffmpeg, &layered);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame * 2, "ffmpeg picture count");
        let ffmpeg_base = &raw[frame..];
        assert_eq!(ffmpeg_base, &planes_of(&base));
        assert_ne!(ffmpeg_base, &planes_of(&picture));
    }
}

#[test]
fn inter_layer_residual_follows_the_base_motion() {
    let idr = residual_idr();
    let mut decoder = open();
    let first = decode_with(&mut decoder, &idr);
    assert!(split_luma(&first, 100, 180));
    let base = decode_with(&mut decoder, &moving_base_packet());
    assert!(shifted_residual_picture(&base));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &idr);
    let picture = decode_with(&mut decoder, &moving_residual_packet());
    assert!(base_residual_picture(&picture));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut layered = idr.clone();
        layered.extend(moving_residual_packet());
        let raw = ffmpeg_raw(&ffmpeg, &layered);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame * 2, "ffmpeg picture count");
        let ffmpeg_base = &raw[frame..];
        assert_eq!(ffmpeg_base, &planes_of(&base));
        assert_ne!(ffmpeg_base, &planes_of(&picture));
    }
}

#[test]
fn inter_layer_base_mode_adds_the_residual_and_its_own() {
    let idr = residual_idr();
    let mut decoder = open();
    let first = decode_with(&mut decoder, &idr);
    assert!(split_luma(&first, 100, 180));
    let base = decode_with(&mut decoder, &moving_base_packet());
    assert!(shifted_residual_picture(&base));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &idr);
    let picture = decode_with(&mut decoder, &both_tools_packet());
    assert!(
        both_tools_picture(&picture),
        "row {:?}",
        &picture.picture.y()[..16]
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut layered = idr.clone();
        layered.extend(both_tools_packet());
        let raw = ffmpeg_raw(&ffmpeg, &layered);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame * 2, "ffmpeg picture count");
        let ffmpeg_base = &raw[frame..];
        assert_eq!(ffmpeg_base, &planes_of(&base));
        assert_ne!(ffmpeg_base, &planes_of(&picture));
    }
}

#[test]
fn inter_layer_prediction_uses_the_older_reference() {
    let older = older_idr();
    let recent = recent_reference();
    let mut decoder = open();
    let first = decode_with(&mut decoder, &older);
    assert!(split_luma(&first, 100, 180));
    let second = decode_with(&mut decoder, &recent);
    assert!(painted(&second, 40, 90));
    let base = decode_with(&mut decoder, &older_base_packet());
    assert!(base_residual_picture(&base));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &older);
    let _ = decode_with(&mut decoder, &recent);
    let inherited = decode_with(&mut decoder, &older_inherit_packet());
    assert!(split_luma(&inherited, 100, 180));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &older);
    let _ = decode_with(&mut decoder, &recent);
    let added = decode_with(&mut decoder, &older_residual_packet());
    assert!(
        corner_on_flat(&added, 43, 40, 90),
        "row {:?}",
        &added.picture.y()[..16]
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut layered = older.clone();
        layered.extend(recent);
        layered.extend(older_residual_packet());
        let raw = ffmpeg_raw(&ffmpeg, &layered);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame * 3, "ffmpeg picture count");
        assert_eq!(&raw[frame * 2..], &planes_of(&base));
        assert_ne!(&raw[frame * 2..], &planes_of(&added));
    }
}

#[test]
fn inter_layer_prediction_uses_list_one() {
    let past = older_idr();
    let future = future_reference();
    let mut decoder = open();
    let first = decode_with(&mut decoder, &past);
    assert!(split_luma(&first, 100, 180));
    let second = decode_with(&mut decoder, &future);
    assert!(painted(&second, 40, 90));
    let base = decode_with(&mut decoder, &list1_base_packet());
    assert!(corner_on_flat(&base, 43, 40, 90));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &past);
    let _ = decode_with(&mut decoder, &future);
    let inherited = decode_with(&mut decoder, &list1_inherit_packet());
    assert!(painted(&inherited, 40, 90));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &past);
    let _ = decode_with(&mut decoder, &future);
    let added = decode_with(&mut decoder, &list1_residual_packet());
    assert!(base_residual_picture(&added));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut layered = past.clone();
        layered.extend(future);
        layered.extend(list1_residual_packet());
        let raw = ffmpeg_raw(&ffmpeg, &layered);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame * 3, "ffmpeg picture count");
        assert_eq!(&raw[frame..frame * 2], &planes_of(&base));
        assert_ne!(&raw[frame..frame * 2], &planes_of(&added));
    }
}

#[test]
fn inter_layer_prediction_averages_both_lists() {
    let past = older_idr();
    let future = future_reference();
    let mut decoder = open();
    let first = decode_with(&mut decoder, &past);
    assert!(split_luma(&first, 100, 180));
    let second = decode_with(&mut decoder, &future);
    assert!(painted(&second, 40, 90));
    let base = decode_with(&mut decoder, &bipred_base_packet());
    assert!(corner_on_split(&base, 73, 70, 110, 90));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &past);
    let _ = decode_with(&mut decoder, &future);
    let inherited = decode_with(&mut decoder, &bipred_inherit_packet());
    assert!(split_luma(&inherited, 70, 110));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &past);
    let _ = decode_with(&mut decoder, &future);
    let added = decode_with(&mut decoder, &bipred_residual_packet());
    assert!(base_residual_picture(&added));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut layered = past.clone();
        layered.extend(future);
        layered.extend(bipred_residual_packet());
        let raw = ffmpeg_raw(&ffmpeg, &layered);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame * 3, "ffmpeg picture count");
        assert_eq!(&raw[frame..frame * 2], &planes_of(&base));
        assert_ne!(&raw[frame..frame * 2], &planes_of(&added));
    }
}

#[test]
fn inter_view_prediction_uses_the_base_view() {
    let parsed = parse_subset_sps(&mvc_views(118, false, true)).expect("mvc");
    assert_eq!(parsed.views, vec![0, 1]);
    assert_eq!(parsed.interview[1].anchor_l0, vec![0]);
    assert!(parsed.interview[1].anchor_l1.is_empty());
    let mut decoder = open();
    let base = decode_with(&mut decoder, &interview_base());
    assert!(split_luma(&base, 100, 180));
    let newer = decode_with(&mut decoder, &future_reference());
    assert!(painted(&newer, 40, 90));
    let view = decode_with(&mut decoder, &interview_view());
    assert!(
        base_residual_picture(&view),
        "row {:?}",
        &view.picture.y()[..16]
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = interview_base();
        annex.extend(future_reference());
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame * 2, "ffmpeg picture count");
        assert_eq!(&raw[..frame], &planes_of(&base));
        assert_ne!(&raw[..frame], &planes_of(&view));
    }
}

#[test]
fn inter_layer_motion_adds_an_8x8_residual() {
    let mut ordinary = open();
    let idr = decode_with(&mut ordinary, &eight_idr());
    assert!(painted(&idr, 150, 90));
    let predicted = decode_with(&mut ordinary, &annex_b_nal(0, 1, &eight_p_slice()));
    assert_ne!(planes_of(&predicted), planes_of(&idr));
    assert_eq!(predicted.picture.y()[8 * 16], 150);
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = eight_idr();
        annex.extend(annex_b_nal(0, 1, &eight_p_slice()));
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame * 2, "ffmpeg picture count");
        assert_eq!(&raw[frame..], &planes_of(&predicted));
    }
    let mut layered = open();
    let _ = decode_with(&mut layered, &eight_idr());
    let picture = decode_with(&mut layered, &eight_layer());
    assert_eq!(planes_of(&picture), planes_of(&predicted));
}

#[test]
fn inter_layer_intra_adds_an_8x8_residual() {
    let mut ordinary = open();
    let idr = decode_with(&mut ordinary, &intra8_idr());
    assert!(painted(&idr, 200, 40));
    let predicted = decode_with(&mut ordinary, &annex_b_nal(0, 1, &eight_p_slice()));
    assert_ne!(planes_of(&predicted), planes_of(&idr));
    assert_eq!(predicted.picture.y()[8 * 16], 200);
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = intra8_idr();
        annex.extend(annex_b_nal(0, 1, &eight_p_slice()));
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame * 2, "ffmpeg picture count");
        assert_eq!(&raw[frame..], &planes_of(&predicted));
    }
    let picture = decode(&intra8_layer());
    assert_eq!(planes_of(&picture), planes_of(&predicted));
    assert!(picture.picture.u().iter().all(|sample| *sample == 40));
}

#[test]
fn adaptive_base_mode_chooses_per_macroblock() {
    let picture = decode(&adaptive_annex());
    assert_eq!(picture.picture.width(), 32);
    let y = picture.picture.y();
    for row in 0..16 {
        for col in 0..32 {
            let want = if col < 16 { 100 } else { 40 };
            assert_eq!(y[row * 32 + col], want, "{row},{col}");
        }
    }
    for plane in [picture.picture.u(), picture.picture.v()] {
        for row in 0..8 {
            for col in 0..16 {
                let want = if col < 8 { 90 } else { 40 };
                assert_eq!(plane[row * 16 + col], want, "{row},{col}");
            }
        }
    }
    if let Some(ffmpeg) = ffmpeg_bin() {
        let raw = ffmpeg_raw(&ffmpeg, &adaptive_annex());
        assert_eq!(raw.len(), 32 * 16 + 16 * 8 * 2);
        assert_ne!(raw, planes_of(&picture));
        assert_eq!(&raw[..32], &paint_row(100, 180));
    }
}

#[test]
fn adaptive_residual_prediction_chooses_per_macroblock() {
    let idr = residual_flag_idr();
    let mut decoder = open();
    let first = decode_with(&mut decoder, &idr);
    assert_eq!(first.picture.width(), 32);
    assert!(painted(&first, 150, 90));
    let base = decode_with(&mut decoder, &residual_flag_base());
    assert!(
        residual_corners(&base, 153, 153, 150),
        "base row {:?}",
        &base.picture.y()[..32]
    );
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &idr);
    let picture = decode_with(&mut decoder, &residual_flag_layer());
    assert!(
        residual_corners(&picture, 153, 150, 150),
        "row {:?}",
        &picture.picture.y()[..32]
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut layered = idr;
        layered.extend(residual_flag_layer());
        let raw = ffmpeg_raw(&ffmpeg, &layered);
        let frame = 32 * 16 + 16 * 8 * 2;
        assert_eq!(raw.len(), frame * 2, "ffmpeg picture count");
        assert_eq!(&raw[frame..], &planes_of(&base));
        assert_ne!(&raw[frame..], &planes_of(&picture));
    }
}

#[test]
fn motion_prediction_flag_uses_the_reference_vector() {
    let idr = motion_flag_idr();
    let mut decoder = open();
    let first = decode_with(&mut decoder, &idr);
    assert!(striped(&first, 100, 180));
    let base = decode_with(&mut decoder, &motion_flag_base());
    assert!(
        shifted_pair(&base),
        "base row {:?}",
        &base.picture.y()[..32]
    );
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &idr);
    let picture = decode_with(&mut decoder, &motion_flag_layer());
    assert!(
        motion_flag_picture(&picture),
        "row {:?}",
        &picture.picture.y()[..32]
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut layered = idr;
        layered.extend(motion_flag_layer());
        let raw = ffmpeg_raw(&ffmpeg, &layered);
        let frame = 32 * 16 + 16 * 8 * 2;
        assert_eq!(raw.len(), frame * 2, "ffmpeg picture count");
        assert_eq!(&raw[frame..], &planes_of(&base));
        assert_ne!(&raw[frame..], &planes_of(&picture));
    }
}

#[test]
fn intra_resampling_upsamples_the_base_layer() {
    let parsed = parse_subset_sps(&spatial_subset()).expect("subset");
    assert_eq!(parsed.svc_spatial, 1);
    assert_eq!(parsed.width, 32);
    assert_eq!(parsed.height, 32);
    assert_eq!(
        (
            parsed.svc_scale.left,
            parsed.svc_scale.top,
            parsed.svc_scale.right,
            parsed.svc_scale.bottom,
        ),
        (0, 0, 0, 0)
    );
    assert_eq!(parsed.svc_scale.chroma_x, 0);
    assert_eq!(parsed.svc_scale.chroma_y, 0);
    assert_eq!(parsed.svc_scale.ref_chroma_x, 0);
    assert_eq!(parsed.svc_scale.ref_chroma_y, 0);
    let base = decode(&resample_base());
    assert!(
        resample_base_picture(&base),
        "base {:?}",
        &base.picture.y()[..16]
    );
    let picture = decode(&resample_annex());
    assert_eq!(picture.picture.width(), 32);
    assert_eq!(picture.picture.height(), 32);
    assert!(
        resampled(&picture),
        "row {:?} chroma {:?}",
        &picture.picture.y()[..32],
        &picture.picture.u()[..16]
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let raw = ffmpeg_raw(&ffmpeg, &resample_annex());
        assert_eq!(raw, planes_of(&base));
        assert_ne!(raw, planes_of(&picture));
    }
}

#[test]
fn motion_resampling_scales_the_base_vector() {
    let mut decoder = open();
    let mut pictures = push_pictures(&mut decoder, &motion_scale_idr());
    pictures.extend(push_pictures(&mut decoder, &motion_scale_p()));
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 2);
    let reference = &pictures[0];
    assert_eq!(reference.picture.width(), 32);
    assert_eq!(reference.picture.height(), 32);
    assert!(
        upsampled_reference(reference),
        "row {:?} chroma {:?}",
        &reference.picture.y()[..32],
        &reference.picture.u()[..16]
    );
    let picture = &pictures[1];
    assert_eq!(picture.picture.width(), 32);
    assert_eq!(picture.picture.height(), 32);
    assert!(
        scaled_motion_picture(picture),
        "row {:?} chroma {:?}",
        &picture.picture.y()[..32],
        &picture.picture.u()[..16]
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = motion_scale_idr();
        annex.extend(motion_scale_p());
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        let base = decode(&resample_base());
        let mut flat = vec![200u8; 16 * 16];
        flat.extend(std::iter::repeat_n(200, 8 * 8 * 2));
        let mut want = planes_of(&base);
        want.extend(flat);
        assert_eq!(raw, want);
        assert_ne!(planes_of(picture), &raw[raw.len() / 2..]);
    }
}

#[test]
fn residual_resampling_upsamples_the_base_difference() {
    let mut decoder = open();
    let mut pictures = push_pictures(&mut decoder, &motion_scale_idr());
    pictures.extend(push_pictures(&mut decoder, &residual_scale_p()));
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 2);
    assert!(upsampled_reference(&pictures[0]));
    let picture = &pictures[1];
    assert_eq!(picture.picture.width(), 32);
    assert_eq!(picture.picture.height(), 32);
    assert!(
        residual_resampled(picture),
        "row {:?} row8 {:?} chroma {:?}",
        &picture.picture.y()[..32],
        &picture.picture.y()[8 * 32..9 * 32],
        &picture.picture.u()[..16]
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = motion_scale_idr();
        annex.extend(residual_scale_p());
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        let base = decode(&resample_base());
        let mut second = planes_of(&base);
        for row in 0..4 {
            for col in 0..4 {
                second[row * 16 + col] = 43;
            }
        }
        let mut want = planes_of(&base);
        want.extend(second);
        assert_eq!(raw, want);
        assert_ne!(planes_of(picture), &raw[raw.len() / 2..]);
    }
}

#[test]
fn a_crop_scales_motion_inside_the_window() {
    let mut decoder = open();
    let mut pictures = push_pictures(&mut decoder, &crop_idr());
    pictures.extend(push_pictures(&mut decoder, &crop_p()));
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 2);
    let reference = &pictures[0];
    assert_eq!(reference.picture.width(), 48);
    assert_eq!(reference.picture.height(), 16);
    assert!(
        cropped_reference(reference),
        "row {:?} chroma {:?}",
        &reference.picture.y()[..48],
        &reference.picture.u()[..24]
    );
    let picture = &pictures[1];
    assert!(
        cropped_motion(picture),
        "row {:?} chroma {:?}",
        &picture.picture.y()[..48],
        &picture.picture.u()[..24]
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = crop_idr();
        annex.extend(crop_p());
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        let base = decode(&resample_base());
        let mut flat = vec![200u8; 16 * 16];
        flat.extend(std::iter::repeat_n(200, 8 * 8 * 2));
        let mut want = planes_of(&base);
        want.extend(flat);
        assert_eq!(raw, want);
        assert_ne!(planes_of(picture), &raw[raw.len() / 2..]);
    }
}

#[test]
fn residual_resampling_respects_the_crop() {
    let mut decoder = open();
    let mut pictures = push_pictures(&mut decoder, &crop_idr());
    pictures.extend(push_pictures(&mut decoder, &crop_residual_p()));
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 2);
    assert!(cropped_reference(&pictures[0]));
    let picture = &pictures[1];
    assert!(
        cropped_residual(picture),
        "row {:?} row4 {:?} chroma {:?}",
        &picture.picture.y()[..48],
        &picture.picture.y()[4 * 48..5 * 48],
        &picture.picture.u()[..24]
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = crop_idr();
        annex.extend(crop_residual_p());
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        let base = decode(&resample_base());
        let mut second = planes_of(&base);
        for row in 0..4 {
            for col in 0..4 {
                second[row * 16 + col] = 43;
            }
        }
        let mut want = planes_of(&base);
        want.extend(second);
        assert_eq!(raw, want);
        assert_ne!(planes_of(picture), &raw[raw.len() / 2..]);
    }
}

#[test]
fn a_negative_offset_shifts_the_scaled_window() {
    let mut decoder = open();
    let mut pictures = push_pictures(&mut decoder, &shifted_idr());
    pictures.extend(push_pictures(&mut decoder, &shifted_residual_p()));
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 2);
    let reference = &pictures[0];
    assert_eq!(reference.picture.width(), 32);
    assert!(
        shifted_reference(reference),
        "row {:?} chroma {:?}",
        &reference.picture.y()[..32],
        &reference.picture.u()[..16]
    );
    let picture = &pictures[1];
    assert!(
        shifted_residual(picture),
        "row {:?} row8 {:?} chroma {:?}",
        &picture.picture.y()[..32],
        &picture.picture.y()[8 * 32..9 * 32],
        &picture.picture.u()[..16]
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = shifted_idr();
        annex.extend(shifted_residual_p());
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        let base = decode(&resample_base());
        let mut second = planes_of(&base);
        for row in 0..4 {
            for col in 0..4 {
                second[row * 16 + col] = 43;
            }
        }
        let mut want = planes_of(&base);
        want.extend(second);
        assert_eq!(raw, want);
        assert_ne!(planes_of(picture), &raw[raw.len() / 2..]);
    }
}

#[test]
fn a_reference_crop_changes_the_scaled_vector() {
    let mut decoder = open();
    let mut pictures = push_pictures(&mut decoder, &crop_change_idr());
    pictures.extend(push_pictures(&mut decoder, &crop_change_p()));
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 2);
    let reference = &pictures[0];
    assert!(
        crop_change_reference(reference),
        "row {:?} chroma {:?}",
        &reference.picture.y()[..32],
        &reference.picture.u()[..16]
    );
    let picture = &pictures[1];
    assert!(
        upsampled_reference(picture),
        "row {:?} chroma {:?}",
        &picture.picture.y()[..32],
        &picture.picture.u()[..16]
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = crop_change_idr();
        annex.extend(crop_change_p());
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        let base = decode(&resample_base());
        let mut want = planes_of(&base);
        want.extend(planes_of(&base));
        assert_eq!(raw, want);
        assert_ne!(planes_of(picture), &raw[raw.len() / 2..]);
    }
}

#[test]
fn a_slice_crop_shifts_the_scaled_window() {
    let parsed = parse_subset_sps(&slice_crop_subset()).expect("subset");
    assert_eq!(parsed.svc_spatial, 2);
    assert_eq!(
        (
            parsed.svc_scale.left,
            parsed.svc_scale.top,
            parsed.svc_scale.right,
            parsed.svc_scale.bottom,
        ),
        (0, 0, 0, 0)
    );
    let mut decoder = open();
    let mut pictures = push_pictures(&mut decoder, &slice_crop_idr());
    pictures.extend(push_pictures(&mut decoder, &slice_crop_p()));
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 2);
    let reference = &pictures[0];
    assert!(
        shifted_reference(reference),
        "row {:?} chroma {:?}",
        &reference.picture.y()[..32],
        &reference.picture.u()[..16]
    );
    let picture = &pictures[1];
    assert!(
        shifted_residual(picture),
        "row {:?} row8 {:?} chroma {:?}",
        &picture.picture.y()[..32],
        &picture.picture.y()[8 * 32..9 * 32],
        &picture.picture.u()[..16]
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = slice_crop_idr();
        annex.extend(slice_crop_p());
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        let base = decode(&resample_base());
        let mut second = planes_of(&base);
        for row in 0..4 {
            for col in 0..4 {
                second[row * 16 + col] = 43;
            }
        }
        let mut want = planes_of(&base);
        want.extend(second);
        assert_eq!(raw, want);
        assert_ne!(planes_of(picture), &raw[raw.len() / 2..]);
    }
}

fn push_pictures(decoder: &mut H264Decoder, annex: &[u8]) -> Vec<VideoOutput> {
    decoder
        .push(&Packet::video(annex.to_vec(), 0, 1, 30, true))
        .expect("decode")
}

#[test]
fn a_non_base_mvc_view_is_an_intra_picture() {
    for profile in [118u8, 128] {
        let parsed = parse_subset_sps(&mvc_subset(profile, false)).expect("mvc");
        assert_eq!(parsed.views, vec![0, 1], "profile {profile}");
    }
    let annex = mvc_annex(&mvc_header(1));
    let picture = decode(&annex);
    assert!(flat(&picture));
    assert_eq!(picture.picture.width(), 16);
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &annex).is_empty(),
            "ffmpeg decoded a non-base view"
        );
    }
    let err = open()
        .push(&Packet::video(mvc_annex(&mvc_header(2)), 0, 1, 30, false))
        .expect_err("view");
    assert!(
        matches!(
            err,
            CodecError::Malformed {
                what: "view id",
                ..
            }
        ),
        "{err:?}"
    );
    let vui = parse_subset_sps(&mvc_subset(118, true)).expect_err("vui");
    assert!(matches!(vui, CodecError::Unsupported { what: "mvc vui" }));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &annex);
    assert!(flat(&decode_with(&mut decoder, &base_idr())));
}

#[test]
fn an_annex_i_depth_view_is_an_intra_picture() {
    for profile in [135u8, 138] {
        let parsed = parse_subset_sps(&mvcd_subset(profile, false)).expect("mvcd");
        assert_eq!(parsed.views, vec![0], "profile {profile}");
        assert_eq!(parsed.depth_views, vec![true], "profile {profile}");
    }
    let annex = depth_annex(&mvc_header(0));
    let picture = decode(&annex);
    assert!(flat(&picture));
    assert_eq!(picture.picture.width(), 16);
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &annex).is_empty(),
            "ffmpeg decoded a depth view"
        );
    }
    let err = open()
        .push(&Packet::video(depth_annex(&mvc_header(1)), 0, 1, 30, false))
        .expect_err("view");
    assert!(
        matches!(
            err,
            CodecError::Malformed {
                what: "view id",
                ..
            }
        ),
        "{err:?}"
    );
    let vui = parse_subset_sps(&mvcd_subset(138, true)).expect_err("vui");
    assert!(matches!(vui, CodecError::Unsupported { what: "mvcd vui" }));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &annex);
    assert!(flat(&decode_with(&mut decoder, &base_idr())));
}

#[test]
fn a_3d_avc_header_is_an_intra_picture() {
    let parsed = parse_subset_sps(&three_d_subset(false)).expect("3d");
    assert_eq!(parsed.views, vec![0]);
    assert_eq!(parsed.depth_views, vec![true]);
    assert_eq!(parsed.texture_views, vec![true]);
    assert!(parsed.avc_3d);
    let annex = three_d_annex(&[0x80, 0x42]);
    let picture = decode(&annex);
    assert!(flat(&picture));
    assert_eq!(picture.picture.width(), 16);
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &annex).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
    let err = open()
        .push(&Packet::video(
            three_d_annex(&[0x80, 0xC2]),
            0,
            1,
            30,
            false,
        ))
        .expect_err("view");
    assert!(
        matches!(
            err,
            CodecError::Malformed {
                what: "view id",
                ..
            }
        ),
        "{err:?}"
    );
    let tool = parse_subset_sps(&three_d_subset(true)).expect("prediction");
    assert!(tool.header_prediction);
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &annex);
    assert!(flat(&decode_with(&mut decoder, &base_idr())));
}

#[test]
fn a_predicted_3d_header_copies_the_texture_slice() {
    let texture = base_idr();
    let depth = predicted_depth_annex();
    let mut decoder = open();
    let missing = decoder
        .push(&Packet::video(depth.clone(), 0, 1, 30, false))
        .expect_err("source");
    assert!(
        matches!(missing, CodecError::Malformed { what: "3d", .. }),
        "{missing:?}"
    );
    let mut decoder = open();
    let mut pictures = push_pictures(&mut decoder, &texture);
    pictures.extend(push_pictures(&mut decoder, &depth));
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 2);
    assert!(flat(&pictures[0]));
    assert!(copied_depth(&pictures[1]));
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert_eq!(ffmpeg_bytes(&ffmpeg, &texture), paint(16, 128, 128));
        assert!(
            ffmpeg_bytes(&ffmpeg, &depth).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
    let mut inter = Writer::new();
    inter.ue(0);
    inter.ue(0);
    inter.ue(0);
    inter.trailing();
    let err = decoder
        .push(&Packet::video(
            annex_b_extension(1, 21, &[0x80, 0x60], &inter.bytes),
            0,
            1,
            30,
            false,
        ))
        .expect_err("inter");
    assert!(
        matches!(err, CodecError::Unsupported { what: "3d" }),
        "{err:?}"
    );
}

#[test]
fn a_depth_motion_vector_shifts_the_other_view() {
    let parsed = parse_subset_sps(&three_d_two_views()).expect("views");
    assert_eq!(parsed.views, vec![0, 1]);
    assert_eq!(parsed.max_num_ref_frames, 4);
    let base_sps = crate::set::parse_sps(&four_ref_sps()).expect("base sps");
    assert_eq!(base_sps.max_num_ref_frames, 4);
    let base_idr = dmvp_base_idr();
    let other_idr = dmvp_other_idr();
    let base_p = dmvp_base_p();
    let skip = dmvp_skip(true);
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base_idr, &other_idr, &base_p, &skip] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 4, "picture count");
    assert!(pictures.iter().any(|picture| split_luma(picture, 100, 180)));
    assert!(pictures.iter().any(copied_depth));
    assert!(pictures.iter().any(shifted_residual_picture));
    assert!(pictures.iter().any(|picture| painted(picture, 220, 50)));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = base_idr.clone();
        annex.extend(&base_p);
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame * 2, "ffmpeg picture count");
        let idr = pictures
            .iter()
            .find(|picture| split_luma(picture, 100, 180))
            .expect("base idr");
        let predicted = pictures
            .iter()
            .find(|picture| shifted_residual_picture(picture))
            .expect("base motion");
        assert_eq!(&raw[..frame], &planes_of(idr));
        assert_eq!(&raw[frame..], &planes_of(predicted));
        assert!(
            ffmpeg_bytes(&ffmpeg, &skip).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
    let mut decoder = open();
    for annex in [&base_idr, &other_idr, &base_p] {
        let _ = push_pictures(&mut decoder, annex);
    }
    let err = decoder
        .push(&Packet::video(dmvp_skip(false), 0, 1, 30, false))
        .expect_err("flag");
    assert!(
        matches!(err, CodecError::Unsupported { what: "3d" }),
        "{err:?}"
    );
}

#[test]
fn a_coded_depth_motion_vector_cancels_the_base_motion() {
    let base_idr = dmvp_base_idr();
    let other_idr = dmvp_other_idr();
    let base_p = dmvp_base_p();
    let coded = dmvp_coded();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base_idr, &other_idr, &base_p, &coded] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 4, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| copied_depth(picture))
            .count(),
        2
    );
    assert!(pictures.iter().any(shifted_residual_picture));
    assert!(pictures.iter().all(|picture| !painted(picture, 220, 50)));
    assert!(pictures.iter().all(|picture| !painted(picture, 40, 50)));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = base_idr.clone();
        annex.extend(&base_p);
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame * 2, "ffmpeg picture count");
        let predicted = pictures
            .iter()
            .find(|picture| shifted_residual_picture(picture))
            .expect("base motion");
        assert_eq!(&raw[frame..], &planes_of(predicted));
        assert!(
            ffmpeg_bytes(&ffmpeg, &coded).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn a_b_slice_depth_motion_vector_cancels_the_base_motion() {
    let base_idr = dmvp_base_idr();
    let other_idr = dmvp_other_idr();
    let base_p = dmvp_base_p();
    let coded = dmvp_coded_b(true);
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base_idr, &other_idr, &base_p, &coded] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 4, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| copied_depth(picture))
            .count(),
        2
    );
    assert!(pictures.iter().any(shifted_residual_picture));
    assert!(pictures.iter().all(|picture| !painted(picture, 220, 50)));
    assert!(pictures.iter().all(|picture| !painted(picture, 40, 50)));
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &coded).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
    let mut decoder = open();
    for annex in [&base_idr, &other_idr, &base_p] {
        let _ = push_pictures(&mut decoder, annex);
    }
    let err = decoder
        .push(&Packet::video(dmvp_coded_b(false), 0, 1, 30, false))
        .expect_err("flag");
    assert!(
        matches!(err, CodecError::Unsupported { what: "3d" }),
        "{err:?}"
    );
}

#[test]
fn a_view_synthesis_disparity_shifts_the_base_texture() {
    let base = dmvp_base_idr();
    let depth = vsp_depth();
    let view = vsp_texture();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base, &depth, &view] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 3, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| painted(picture, 180, 90))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| split_luma(picture, 100, 180))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| painted(picture, 128, 40))
            .count(),
        1
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let raw = ffmpeg_raw(&ffmpeg, &base);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame, "ffmpeg picture count");
        let decoded = pictures
            .iter()
            .find(|picture| split_luma(picture, 100, 180))
            .expect("base");
        assert_eq!(raw, planes_of(decoded));
        assert!(
            ffmpeg_bytes(&ffmpeg, &view).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
    let mut decoder = open();
    let _ = push_pictures(&mut decoder, &base);
    let err = decoder
        .push(&Packet::video(vsp_without_depth(), 0, 1, 30, false))
        .expect_err("depth");
    assert!(
        matches!(err, CodecError::Malformed { what: "3d", .. }),
        "{err:?}"
    );
}

#[test]
fn a_far_depth_sample_scales_the_view_synthesis_disparity() {
    let base = dmvp_base_idr();
    let depth = vsp_scaled_depth();
    let view = vsp_texture();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base, &depth, &view] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 3, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| four_pixel_shift(picture))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| split_luma(picture, 100, 180))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| depth_corner(picture))
            .count(),
        1
    );
    assert!(pictures.iter().all(|picture| !painted(picture, 180, 90)));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let raw = ffmpeg_raw(&ffmpeg, &base);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame, "ffmpeg picture count");
        let decoded = pictures
            .iter()
            .find(|picture| split_luma(picture, 100, 180))
            .expect("base");
        assert_eq!(raw, planes_of(decoded));
        assert!(
            ffmpeg_bytes(&ffmpeg, &view).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn a_wide_view_synthesis_picture_samples_each_macroblock() {
    let base = wide_base();
    let depth = wide_vsp_depth();
    let view = wide_vsp_texture();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base, &depth, &view] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 3, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| left_block_shifts(picture))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| wide_split(picture))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| wide_depth_column(picture))
            .count(),
        1
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        let raw = ffmpeg_raw(&ffmpeg, &base);
        let frame = 32 * 16 + 16 * 8 * 2;
        assert_eq!(raw.len(), frame, "ffmpeg picture count");
        let decoded = pictures
            .iter()
            .find(|picture| wide_split(picture))
            .expect("base");
        assert_eq!(raw, planes_of(decoded));
        assert!(
            ffmpeg_bytes(&ffmpeg, &view).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn a_skipped_b_view_synthesis_macroblock_uses_four_8x8_disparities() {
    let base = dmvp_base_idr();
    let depth = vsp_block_depth();
    let view = vsp_b_skip_texture();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base, &depth, &view] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 3, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| top_left_block_shift(picture))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| split_luma(picture, 100, 180))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| depth_block(picture))
            .count(),
        1
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &view).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn a_skipped_view_synthesis_macroblock_uses_four_8x8_disparities() {
    let base = dmvp_base_idr();
    let depth = vsp_block_depth();
    let view = vsp_skip_texture();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base, &depth, &view] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 3, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| top_left_block_shift(picture))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| split_luma(picture, 100, 180))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| depth_block(picture))
            .count(),
        1
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &view).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn a_4x4_view_synthesis_sub_partition_uses_the_8x8_corners() {
    let base = dmvp_base_idr();
    let depth = vsp_block_depth();
    let view = vsp_4x4_texture();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base, &depth, &view] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 3, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| top_left_block_shift(picture))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| split_luma(picture, 100, 180))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| depth_block(picture))
            .count(),
        1
    );
    assert!(pictures.iter().all(|picture| !corner_4x4_only(picture)));
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &view).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn a_4x8_view_synthesis_sub_partition_uses_the_8x8_corners() {
    let base = dmvp_base_idr();
    let depth = vsp_edge_depth();
    let view = vsp_4x8_texture();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base, &depth, &view] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 3, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| top_left_block_shift(picture))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| split_luma(picture, 100, 180))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| depth_edge(picture))
            .count(),
        1
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &view).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn an_8x4_view_synthesis_sub_partition_uses_the_8x8_corners() {
    let base = dmvp_base_idr();
    let depth = vsp_block_depth();
    let view = vsp_8x4_texture();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base, &depth, &view] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 3, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| top_left_block_shift(picture))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| split_luma(picture, 100, 180))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| depth_block(picture))
            .count(),
        1
    );
    assert!(
        pictures
            .iter()
            .all(|picture| !upper_8x4_stays_split(picture))
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &view).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn an_8x8_view_synthesis_block_shifts_only_its_own_corner() {
    let base = dmvp_base_idr();
    let depth = vsp_block_depth();
    let view = vsp_8x8_texture();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base, &depth, &view] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 3, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| top_left_block_shift(picture))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| split_luma(picture, 100, 180))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| depth_block(picture))
            .count(),
        1
    );
    assert!(pictures.iter().all(|picture| !four_pixel_shift(picture)));
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &view).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn an_8x16_view_synthesis_partition_shifts_the_left_half() {
    let base = dmvp_base_idr();
    let depth = vsp_column_depth();
    let view = vsp_8x16_texture();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base, &depth, &view] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 3, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| four_pixel_shift(picture))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| split_luma(picture, 100, 180))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| depth_column(picture))
            .count(),
        1
    );
    assert!(pictures.iter().all(|picture| !near_half_still(picture)));
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &view).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn a_16x8_view_synthesis_partition_keeps_the_near_half_still() {
    let base = dmvp_base_idr();
    let depth = vsp_scaled_depth();
    let view = vsp_16x8_texture();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base, &depth, &view] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 3, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| near_half_still(picture))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| split_luma(picture, 100, 180))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| depth_corner(picture))
            .count(),
        1
    );
    assert!(pictures.iter().all(|picture| !four_pixel_shift(picture)));
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &view).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn a_b_slice_view_synthesis_copies_the_base_texture() {
    let base = dmvp_base_idr();
    let depth = vsp_depth();
    let view = vsp_b_texture();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base, &depth, &view] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 3, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| painted(picture, 180, 90))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| split_luma(picture, 100, 180))
            .count(),
        1
    );
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| painted(picture, 128, 40))
            .count(),
        1
    );
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &view).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn an_8x8_depth_motion_vector_cancels_the_base_motion() {
    let base_idr = dmvp_base_idr();
    let other_idr = dmvp_other_idr();
    let base_p = dmvp_base_p();
    let coded = dmvp_coded_8x8();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base_idr, &other_idr, &base_p, &coded] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 4, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| copied_depth(picture))
            .count(),
        2
    );
    assert!(pictures.iter().any(shifted_residual_picture));
    assert!(pictures.iter().all(|picture| !painted(picture, 220, 50)));
    assert!(pictures.iter().all(|picture| !painted(picture, 40, 50)));
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &coded).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn an_8x4_depth_motion_vector_reads_the_scanned_block() {
    let base_idr = dmvp_base_idr();
    let other_idr = dmvp_other_idr();
    let base_p = dmvp_base_8x4();
    let coded = dmvp_coded_8x4();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base_idr, &other_idr, &base_p, &coded] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 4, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| copied_depth(picture))
            .count(),
        2
    );
    assert!(pictures.iter().any(top_left_band));
    assert!(pictures.iter().all(|picture| !painted(picture, 220, 50)));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = base_idr.clone();
        annex.extend(&base_p);
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        let frame = 16 * 16 + 8 * 8 * 2;
        assert_eq!(raw.len(), frame * 2, "ffmpeg picture count");
        let predicted = pictures
            .iter()
            .find(|picture| top_left_band(picture))
            .expect("base motion");
        assert_eq!(&raw[frame..], &planes_of(predicted));
        assert!(
            ffmpeg_bytes(&ffmpeg, &coded).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn a_4x4_depth_motion_vector_cancels_the_base_motion() {
    let base_idr = dmvp_base_idr();
    let other_idr = dmvp_other_idr();
    let base_p = dmvp_base_p();
    let coded = dmvp_coded_4x4();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base_idr, &other_idr, &base_p, &coded] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 4, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| copied_depth(picture))
            .count(),
        2
    );
    assert!(pictures.iter().any(shifted_residual_picture));
    assert!(pictures.iter().all(|picture| !painted(picture, 220, 50)));
    assert!(pictures.iter().all(|picture| !painted(picture, 40, 50)));
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &coded).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

#[test]
fn a_4x8_depth_motion_vector_cancels_the_base_motion() {
    let base_idr = dmvp_base_idr();
    let other_idr = dmvp_other_idr();
    let base_p = dmvp_base_p();
    let coded = dmvp_coded_4x8();
    let mut decoder = open();
    let mut pictures = Vec::new();
    for annex in [&base_idr, &other_idr, &base_p, &coded] {
        pictures.extend(push_pictures(&mut decoder, annex));
    }
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 4, "picture count");
    assert_eq!(
        pictures
            .iter()
            .filter(|picture| copied_depth(picture))
            .count(),
        2
    );
    assert!(pictures.iter().any(shifted_residual_picture));
    assert!(pictures.iter().all(|picture| !painted(picture, 220, 50)));
    assert!(pictures.iter().all(|picture| !painted(picture, 40, 50)));
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &coded).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

fn dmvp_base_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &four_ref_sps());
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &residual_pcm()));
    out
}

fn dmvp_base_p() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 1, 1, true, false);
    w.ue(0);
    w.ue(0);
    w.se(32);
    w.se(0);
    w.ue(2);
    w.se(0);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.trailing();
    annex_b_nal(2, 1, &w.bytes)
}

fn dmvp_other_idr() -> Vec<u8> {
    let mut w = Writer::new();
    // Not an IDR. An IDR would drop the base view's reference.
    w.ue(0);
    w.ue(2);
    w.ue(0);
    w.bits(0, 4);
    w.bits(0, 4);
    w.bit(false);
    w.se(0);
    w.ue(1);
    w.ue(25);
    w.align_zeros();
    for _row in 0..16 {
        for col in 0..16 {
            w.bytes.push(if col < 8 { 40 } else { 220 });
        }
    }
    w.bytes.extend(std::iter::repeat_n(50, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    let mut out = annex_b_nal(3, 15, &three_d_two_views());
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(3, 21, &[0x80, 0xA0], &w.bytes));
    out
}

fn dmvp_skip(flag: bool) -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 1, 1, true, false);
    w.bit(flag);
    if flag {
        w.ue(0);
    }
    w.ue(1);
    w.trailing();
    annex_b_extension(2, 21, &[0x80, 0xA0], &w.bytes)
}

fn dmvp_b_header(w: &mut Writer, frame_num: u32, poc: u32) {
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.bits(frame_num, 4);
    w.bits(poc, 4);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.se(0);
    w.ue(1);
}

fn vsp_depth() -> Vec<u8> {
    vsp_depth_annex(&view_synthesis_sps(), &vec![128; 256])
}

fn vsp_scaled_depth() -> Vec<u8> {
    let mut luma = vec![0; 256];
    luma[255] = 64;
    vsp_depth_annex(&view_synthesis_pair(0, 0, 3968, 0), &luma)
}

fn vsp_column_depth() -> Vec<u8> {
    let mut luma = vec![0; 256];
    luma[254] = 64;
    vsp_depth_annex(&view_synthesis_pair(0, 0, 3968, 0), &luma)
}

fn vsp_block_depth() -> Vec<u8> {
    let mut luma = vec![0; 256];
    luma[14 * 16 + 14] = 64;
    vsp_depth_annex(&view_synthesis_pair(0, 0, 3968, 0), &luma)
}

fn vsp_edge_depth() -> Vec<u8> {
    let mut luma = vec![0; 256];
    luma[14 * 16] = 64;
    vsp_depth_annex(&view_synthesis_pair(0, 0, 3968, 0), &luma)
}

fn vsp_depth_annex(sps: &[u8], luma: &[u8]) -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, sps);
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(
        3,
        21,
        &[0x80, 0x60],
        &vsp_depth_slice(luma),
    ));
    out
}

fn vsp_texture() -> Vec<u8> {
    annex_b_extension(2, 21, &[0x80, 0xA0], &vsp_p_slice())
}

fn vsp_b_skip_texture() -> Vec<u8> {
    annex_b_extension(2, 21, &[0x80, 0xA0], &vsp_b_skip_slice())
}

fn vsp_b_skip_slice() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.bits(0, 4);
    w.bits(0, 4);
    w.bit(true);
    w.bit(true);
    w.ue(1);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.se(0);
    w.ue(1);
    w.bit(false);
    w.bit(true);
    w.ue(1);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn vsp_skip_texture() -> Vec<u8> {
    annex_b_extension(2, 21, &[0x80, 0xA0], &vsp_skip_slice())
}

fn vsp_skip_slice() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 0, 0, true, true);
    w.bit(false);
    w.bit(true);
    w.ue(1);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn vsp_4x4_texture() -> Vec<u8> {
    annex_b_extension(2, 21, &[0x80, 0xA0], &vsp_4x4_slice())
}

fn vsp_4x4_slice() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 0, 0, true, true);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(3);
    w.ue(3);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    for _ in 0..4 {
        w.bit(true);
        w.bit(true);
    }
    w.ue(0);
    w.trailing();
    w.bytes
}

fn vsp_4x8_texture() -> Vec<u8> {
    annex_b_extension(2, 21, &[0x80, 0xA0], &vsp_4x8_slice())
}

fn vsp_4x8_slice() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 0, 0, true, true);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(3);
    w.ue(2);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    for _ in 0..4 {
        w.bit(true);
        w.bit(true);
    }
    w.ue(0);
    w.trailing();
    w.bytes
}

fn vsp_8x4_texture() -> Vec<u8> {
    annex_b_extension(2, 21, &[0x80, 0xA0], &vsp_8x4_slice())
}

fn vsp_8x4_slice() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 0, 0, true, true);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(3);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    for _ in 0..4 {
        w.bit(true);
        w.bit(true);
    }
    w.ue(0);
    w.trailing();
    w.bytes
}

fn vsp_8x8_texture() -> Vec<u8> {
    annex_b_extension(2, 21, &[0x80, 0xA0], &vsp_8x8_slice())
}

fn vsp_8x8_slice() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 0, 0, true, true);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(3);
    for _ in 0..4 {
        w.ue(0);
    }
    for _ in 0..4 {
        w.bit(true);
        w.bit(true);
    }
    w.ue(0);
    w.trailing();
    w.bytes
}

fn vsp_8x16_texture() -> Vec<u8> {
    annex_b_extension(2, 21, &[0x80, 0xA0], &vsp_8x16_slice())
}

fn vsp_8x16_slice() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 0, 0, true, true);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(2);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn vsp_16x8_texture() -> Vec<u8> {
    annex_b_extension(2, 21, &[0x80, 0xA0], &vsp_16x8_slice())
}

fn vsp_16x8_slice() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 0, 0, true, true);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(1);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn vsp_without_depth() -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, &view_synthesis_sps());
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(vsp_texture());
    out
}

fn vsp_depth_slice(luma: &[u8]) -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(2);
    w.ue(0);
    w.bits(0, 4);
    w.bits(0, 4);
    w.bit(false);
    w.se(0);
    w.ue(1);
    w.ue(25);
    w.align_zeros();
    w.bytes.extend(luma);
    w.bytes.extend(std::iter::repeat_n(40, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    w.bytes
}

fn near_half_still(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    let top = (0..8).all(|row| {
        let start = row * 16;
        y[start..start + 8].iter().all(|sample| *sample == 100)
            && y[start + 8..start + 16].iter().all(|sample| *sample == 180)
    });
    let bottom = (8..16).all(|row| {
        let start = row * 16;
        y[start..start + 4].iter().all(|sample| *sample == 100)
            && y[start + 4..start + 16].iter().all(|sample| *sample == 180)
    });
    top && bottom
        && picture
            .picture
            .u()
            .iter()
            .chain(picture.picture.v())
            .all(|sample| *sample == 90)
}

fn four_pixel_shift(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    (0..16).all(|row| {
        let start = row * 16;
        y[start..start + 4].iter().all(|sample| *sample == 100)
            && y[start + 4..start + 16].iter().all(|sample| *sample == 180)
    }) && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == 90)
}

fn corner_4x4_only(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    let upper = (0..4).all(|row| {
        let start = row * 16;
        y[start..start + 8].iter().all(|sample| *sample == 100)
            && y[start + 8..start + 16].iter().all(|sample| *sample == 180)
    });
    let lower = (4..8).all(|row| {
        let start = row * 16;
        y[start..start + 4].iter().all(|sample| *sample == 100)
            && y[start + 4..start + 16].iter().all(|sample| *sample == 180)
    });
    let bottom = (8..16).all(|row| {
        let start = row * 16;
        y[start..start + 8].iter().all(|sample| *sample == 100)
            && y[start + 8..start + 16].iter().all(|sample| *sample == 180)
    });
    upper
        && lower
        && bottom
        && picture
            .picture
            .u()
            .iter()
            .chain(picture.picture.v())
            .all(|sample| *sample == 90)
}

fn upper_8x4_stays_split(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    let upper = (0..4).all(|row| {
        let start = row * 16;
        y[start..start + 8].iter().all(|sample| *sample == 100)
            && y[start + 8..start + 16].iter().all(|sample| *sample == 180)
    });
    let lower = (4..8).all(|row| {
        let start = row * 16;
        y[start..start + 4].iter().all(|sample| *sample == 100)
            && y[start + 4..start + 16].iter().all(|sample| *sample == 180)
    });
    let bottom = (8..16).all(|row| {
        let start = row * 16;
        y[start..start + 8].iter().all(|sample| *sample == 100)
            && y[start + 8..start + 16].iter().all(|sample| *sample == 180)
    });
    upper && lower && bottom
}

fn top_left_block_shift(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    let top = (0..8).all(|row| {
        let start = row * 16;
        y[start..start + 4].iter().all(|sample| *sample == 100)
            && y[start + 4..start + 16].iter().all(|sample| *sample == 180)
    });
    let bottom = (8..16).all(|row| {
        let start = row * 16;
        y[start..start + 8].iter().all(|sample| *sample == 100)
            && y[start + 8..start + 16].iter().all(|sample| *sample == 180)
    });
    top && bottom
        && picture
            .picture
            .u()
            .iter()
            .chain(picture.picture.v())
            .all(|sample| *sample == 90)
}

fn depth_edge(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    let at = 14 * 16;
    y.len() == 256
        && y[at] == 64
        && y.iter()
            .enumerate()
            .all(|(index, sample)| index == at || *sample == 0)
        && picture
            .picture
            .u()
            .iter()
            .chain(picture.picture.v())
            .all(|sample| *sample == 40)
}

fn depth_block(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    let at = 14 * 16 + 14;
    y.len() == 256
        && y[at] == 64
        && y.iter()
            .enumerate()
            .all(|(index, sample)| index == at || *sample == 0)
        && picture
            .picture
            .u()
            .iter()
            .chain(picture.picture.v())
            .all(|sample| *sample == 40)
}

fn depth_column(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    y.len() == 256
        && y[254] == 64
        && y.iter()
            .enumerate()
            .all(|(index, sample)| index == 254 || *sample == 0)
        && picture
            .picture
            .u()
            .iter()
            .chain(picture.picture.v())
            .all(|sample| *sample == 40)
}

fn depth_corner(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    y.len() == 256
        && y[255] == 64
        && y[..255].iter().all(|sample| *sample == 0)
        && picture
            .picture
            .u()
            .iter()
            .chain(picture.picture.v())
            .all(|sample| *sample == 40)
}

fn left_block_shifts(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    y.len() == 32 * 16
        && (0..16).all(|row| {
            let start = row * 32;
            let samples = &y[start..start + 32];
            samples[..4].iter().all(|sample| *sample == 100)
                && samples[4..12].iter().all(|sample| *sample == 180)
                && samples[12..24].iter().all(|sample| *sample == 100)
                && samples[24..].iter().all(|sample| *sample == 180)
        })
        && picture
            .picture
            .u()
            .iter()
            .chain(picture.picture.v())
            .all(|sample| *sample == 90)
}

fn wide_split(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    y.len() == 32 * 16
        && (0..16).all(|row| {
            let start = row * 32;
            let samples = &y[start..start + 32];
            samples[..8].iter().all(|sample| *sample == 100)
                && samples[8..16].iter().all(|sample| *sample == 180)
                && samples[16..24].iter().all(|sample| *sample == 100)
                && samples[24..].iter().all(|sample| *sample == 180)
        })
        && picture
            .picture
            .u()
            .iter()
            .chain(picture.picture.v())
            .all(|sample| *sample == 90)
}

fn wide_depth_column(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    let at = 15 * 32 + 30;
    y.len() == 32 * 16
        && y[at] == 64
        && y.iter()
            .enumerate()
            .all(|(index, sample)| index == at || *sample == 0)
        && picture
            .picture
            .u()
            .iter()
            .chain(picture.picture.v())
            .all(|sample| *sample == 40)
}

fn wide_base() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &wide_ref_sps());
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &wide_base_slice()));
    out
}

fn wide_base_slice() -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    split_pcm(&mut w, 90);
    split_pcm(&mut w, 90);
    w.trailing();
    w.bytes
}

fn wide_vsp_depth() -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, &view_synthesis_mbs(0, 0, 3968, 0, 2));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(3, 21, &[0x80, 0x60], &wide_depth_slice()));
    out
}

fn wide_depth_slice() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(2);
    w.ue(0);
    w.bits(0, 4);
    w.bits(0, 4);
    w.bit(false);
    w.se(0);
    w.ue(1);
    zero_pcm(&mut w);
    let mut right = vec![0u8; 256];
    right[15 * 16 + 14] = 64;
    pcm_block(&mut w, &right, 40);
    w.trailing();
    w.bytes
}

fn wide_vsp_texture() -> Vec<u8> {
    annex_b_extension(2, 21, &[0x80, 0xA0], &wide_vsp_slice())
}

fn wide_vsp_slice() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 0, 0, true, true);
    w.bit(false);
    w.bit(true);
    for _ in 0..2 {
        w.ue(0);
        w.ue(0);
        w.bit(true);
        w.bit(true);
        w.ue(0);
    }
    w.trailing();
    w.bytes
}

fn split_pcm(w: &mut Writer, chroma: u8) {
    let mut luma = Vec::with_capacity(256);
    for _row in 0..16 {
        for col in 0..16 {
            luma.push(if col < 8 { 100 } else { 180 });
        }
    }
    pcm_block(w, &luma, chroma);
}

fn zero_pcm(w: &mut Writer) {
    pcm_block(w, &[0; 256], 40);
}

fn pcm_block(w: &mut Writer, luma: &[u8], chroma: u8) {
    w.ue(25);
    w.align_zeros();
    w.bytes.extend(luma);
    w.bytes.extend(std::iter::repeat_n(chroma, 128));
    w.bit = w.bytes.len() * 8;
}

fn vsp_p_slice() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 0, 0, true, true);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn vsp_b_texture() -> Vec<u8> {
    annex_b_extension(2, 21, &[0x80, 0xA0], &vsp_b_slice())
}

fn vsp_b_slice() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.bits(0, 4);
    w.bits(0, 4);
    w.bit(true);
    w.bit(true);
    w.ue(1);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.se(0);
    w.ue(1);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(1);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.trailing();
    w.bytes
}

/// Two paired views. List 0 names view 0. Scale 0 and offset 2016 make disparity 32.
fn view_synthesis_sps() -> Vec<u8> {
    view_synthesis_pair(0, 0, 0, 2016)
}

/// `vsp_param` for the one depth-view pair. `wij` is the scale from view 1 to view 0.
fn view_synthesis_pair(wji: u32, oji: u32, wij: u32, oij: u32) -> Vec<u8> {
    view_synthesis_mbs(wji, oji, wij, oij, 1)
}

fn view_synthesis_mbs(wji: u32, oji: u32, wij: u32, oij: u32, mbs: u32) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(139, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(4);
    w.bit(false);
    w.ue(mbs.saturating_sub(1));
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(1);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.ue(1);
    w.bit(true);
    w.bit(true);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.bits(30, 8);
    w.ue(0);
    w.bits(0, 3);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.ue(2);
    w.ue(0);
    w.ue(1);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    write_depth_value(&mut w, 31);
    w.bit(true);
    write_depth_value(&mut w, 32);
    w.ue(wji);
    w.ue(oji);
    w.ue(wij);
    w.ue(oij);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn dmvp_coded_b(flag: bool) -> Vec<u8> {
    let mut w = Writer::new();
    dmvp_b_header(&mut w, 1, 1);
    w.bit(flag);
    if flag {
        w.ue(0);
        w.ue(0);
        w.ue(1);
        w.se(-32);
        w.se(0);
        w.ue(0);
    }
    w.trailing();
    annex_b_extension(2, 21, &[0x80, 0xA0], &w.bytes)
}

fn dmvp_coded() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 1, 1, true, false);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.se(-32);
    w.se(0);
    w.ue(0);
    w.trailing();
    annex_b_extension(2, 21, &[0x80, 0xA0], &w.bytes)
}

fn dmvp_coded_8x8() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 1, 1, true, false);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.ue(3);
    for _sub in 0..4 {
        w.ue(0);
    }
    // Top row medians are the eight-pixel base vector. Bottom row medians are zero.
    w.se(-32);
    w.se(0);
    w.se(-32);
    w.se(0);
    w.se(0);
    w.se(0);
    w.se(0);
    w.se(0);
    w.ue(0);
    w.trailing();
    annex_b_extension(2, 21, &[0x80, 0xA0], &w.bytes)
}

fn dmvp_coded_4x4() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 1, 1, true, false);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.ue(3);
    for _sub in 0..4 {
        w.ue(3);
    }
    // The top row's median is the eight-pixel base vector. Every other block's is zero.
    for dx in [-32, -32, 0, 0, -32, -32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] {
        w.se(dx);
        w.se(0);
    }
    w.ue(0);
    w.trailing();
    annex_b_extension(2, 21, &[0x80, 0xA0], &w.bytes)
}

fn dmvp_coded_4x8() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 1, 1, true, false);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.ue(3);
    for _sub in 0..4 {
        w.ue(2);
    }
    // The top row's median is the eight-pixel base vector. The bottom row's is zero.
    for dx in [-32, -32, -32, -32, 0, 0, 0, 0] {
        w.se(dx);
        w.se(0);
    }
    w.ue(0);
    w.trailing();
    annex_b_extension(2, 21, &[0x80, 0xA0], &w.bytes)
}

fn dmvp_base_8x4() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 1, 1, true, false);
    w.ue(0);
    w.ue(3);
    for _sub in 0..4 {
        w.ue(1);
    }
    // The second 8×4 moves eight pixels. The block an 8×4 scan would read stays still.
    for dx in [0, 32, 0, 0, 0, 0, 0, 0] {
        w.se(dx);
        w.se(0);
    }
    w.ue(0);
    w.trailing();
    annex_b_nal(2, 1, &w.bytes)
}

fn dmvp_coded_8x4() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 1, 1, true, false);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.ue(3);
    for _sub in 0..4 {
        w.ue(1);
    }
    for _part in 0..8 {
        w.se(0);
        w.se(0);
    }
    w.ue(0);
    w.trailing();
    annex_b_extension(2, 21, &[0x80, 0xA0], &w.bytes)
}

fn top_left_band(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    let luma = y.chunks(16).enumerate().all(|(row, samples)| {
        let left = if (4..8).contains(&row) { 180 } else { 100 };
        samples[..8].iter().all(|sample| *sample == left)
            && samples[8..].iter().all(|sample| *sample == 180)
    });
    luma && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == 90)
}

fn copied_depth(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    let luma = y.chunks(16).all(|row| {
        row[..8].iter().all(|sample| *sample == 40) && row[8..].iter().all(|sample| *sample == 220)
    });
    luma && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == 50)
}

fn predicted_depth_annex() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(2);
    w.ue(0);
    w.bits(2, 2);
    w.bits(2, 2);
    w.se(0);
    w.ue(25);
    w.align_zeros();
    for _row in 0..16 {
        for col in 0..16 {
            w.bytes.push(if col < 8 { 40 } else { 220 });
        }
    }
    w.bytes.extend(std::iter::repeat_n(50, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    let mut out = annex_b_nal(3, 15, &three_d_subset(true));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(3, 21, &[0x80, 0x42], &w.bytes));
    out
}

#[test]
fn a_predicted_3d_depth_picture_copies_its_reference() {
    let idr = three_d_pcm();
    let mut decoder = open();
    let mut pictures = push_pictures(&mut decoder, &idr);
    pictures.extend(push_pictures(&mut decoder, &three_d_skip(false)));
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 2);
    assert!(pictures.iter().all(split_depth));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = idr.clone();
        annex.extend(three_d_skip(false));
        assert!(
            ffmpeg_bytes(&ffmpeg, &annex).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
    let mut decoder = open();
    let held = push_pictures(&mut decoder, &idr);
    assert!(held.is_empty());
    let err = decoder
        .push(&Packet::video(three_d_skip(true), 0, 1, 30, false))
        .expect_err("weighted");
    assert!(matches!(err, CodecError::Unsupported { what: "3d" }));
    let picture = decoder.drain().expect("drain");
    assert_eq!(picture.len(), 1);
    assert!(split_depth(&picture[0]));
    let mut decoder = open();
    assert!(push_pictures(&mut decoder, &idr).is_empty());
    let err = decoder
        .push(&Packet::video(
            annex_b_extension(1, 21, &[0x80, 0x20], &three_d_skip_body(false)),
            0,
            1,
            30,
            false,
        ))
        .expect_err("texture");
    assert!(
        matches!(err, CodecError::Unsupported { what: "3d" }),
        "{err:?}"
    );
}

#[test]
fn an_annex_a_picture_survives_svc_mvc_and_3d_nal_units() {
    let annex = base_with_extensions();
    let picture = decode(&annex);
    assert!(flat(&picture));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        assert_eq!(raw, vec![128u8; 16 * 16 + 8 * 8 * 2]);
    }
}

#[test]
fn the_svc_extension_syntax_is_consumed() {
    let parsed = parse_subset_sps(&subset(83, true, 1, false)).expect("ess");
    assert!(parsed.slice_header_restriction);
    let err = parse_subset_sps(&subset(83, true, 3, false)).expect_err("ess");
    assert!(
        matches!(
            err,
            CodecError::Malformed {
                what: "extended spatial scalability",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn a_refused_extension_leaves_the_next_idr() {
    let mut decoder = open();
    let refused = [
        annex_b_nal(3, 20, &[]),
        annex_b_extension(3, 20, &[0xc0, 0x80, 0x04], &[0x80]),
        svc_annex_header(svc_bytes(false, 0, true)),
        annex_b_extension(3, 20, &[0xc0, 0x80, 0x17], &[0x80]),
        annex_b_extension(3, 20, &[0x00, 0x00, 0x03], &[0x80]),
        annex_b_extension(3, 21, &[0x80, 0x00], &[0x80]),
        annex_b_nal(3, 15, &subset(83, true, 0, true)),
        annex_b_nal(3, 15, &truncated_scalable()),
        stored_base_picture(),
        base_with_broken_layer(),
        residual_layer(),
    ];
    let mut errors = Vec::new();
    for annex in refused {
        errors.push(
            decoder
                .push(&Packet::video(annex, 0, 1, 30, false))
                .expect_err("refused"),
        );
    }
    assert!(matches!(
        errors[0],
        CodecError::Truncated {
            what: "extension header",
            ..
        }
    ));
    assert!(matches!(
        errors[1],
        CodecError::Malformed {
            what: "extension header",
            ..
        }
    ));
    assert!(matches!(
        errors[2],
        CodecError::Unsupported {
            what: "inter-layer prediction"
        }
    ));
    assert!(matches!(
        errors[3],
        CodecError::Unsupported {
            what: "reference base picture"
        }
    ));
    assert!(matches!(errors[4], CodecError::Truncated { .. }));
    assert!(matches!(errors[5], CodecError::Unsupported { what: "3d" }));
    assert!(matches!(
        errors[6],
        CodecError::Unsupported { what: "svc vui" }
    ));
    assert!(matches!(errors[7], CodecError::Truncated { .. }));
    assert!(matches!(
        errors[8],
        CodecError::Unsupported {
            what: "reference base picture"
        }
    ));
    assert!(matches!(errors[9], CodecError::Truncated { .. }));
    assert!(matches!(
        errors[10],
        CodecError::Unsupported {
            what: "inter-layer prediction"
        }
    ));
    let picture = decode_with(&mut decoder, &base_idr());
    assert!(flat(&picture));
}

fn stored_base_picture() -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, &subset(83, false, 0, false));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(
        3,
        20,
        &svc_bytes(true, 0, true),
        &idr_slice(true, false, true),
    ));
    out
}

fn svc_annex(rbsp: &[u8], quality: u8, output: bool, restriction: bool) -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, rbsp);
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(
        3,
        20,
        &svc_bytes(true, quality, output),
        &idr_slice(quality == 0, restriction, false),
    ));
    out
}

fn svc_annex_header(header: [u8; 3]) -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, &subset(83, true, 0, false));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(3, 20, &header, &[0x80]));
    out
}

fn base_with_extensions() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 15, &subset(83, true, 0, false)));
    out.extend(annex_b_nal(3, 15, &baseline_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(3, 14, &svc_bytes(true, 0, true), &[0x80]));
    out.extend(annex_b_nal(3, 5, &idr_slice(true, true, false)));
    out.extend(annex_b_extension(3, 20, &[0x00, 0x00, 0x03], &[0x80]));
    out.extend(annex_b_extension(3, 21, &[0x80, 0x00], &[0x80]));
    out
}

fn base_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &idr_slice(true, true, false)));
    out
}

fn svc_bytes(no_inter_layer_pred: bool, quality: u8, output: bool) -> [u8; 3] {
    let b1 = (u8::from(no_inter_layer_pred) << 7) | (quality & 0x0f);
    let b2 = 0x03 | (u8::from(output) << 2);
    [0xc0, b1, b2]
}

/// IDR view, `view_id` in the low bits, anchor, reserved 1.
fn mvc_header(view: u16) -> [u8; 3] {
    let view = view & 0x03ff;
    let b1 = (view >> 2) as u8;
    let b2 = ((view as u8) << 6) | 0x05;
    [0x00, b1, b2]
}

fn interview_base() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &two_ref_sps());
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &residual_pcm()));
    out
}

fn interview_view() -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, &mvc_views(118, false, true));
    out.extend(annex_b_extension(
        0,
        20,
        &[0x40, 0x00, 0x45],
        &interview_slice(),
    ));
    out
}

fn interview_slice() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 2, 0, false, false);
    w.ue(0);
    w.ue(0);
    w.se(0);
    w.se(0);
    w.ue(2);
    w.se(0);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn mvc_annex(header: &[u8; 3]) -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, &mvc_subset(118, false));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(
        3,
        20,
        header,
        &idr_slice(true, true, false),
    ));
    out
}

fn split_depth(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    let luma = y.chunks(16).all(|row| {
        row[..8].iter().all(|sample| *sample == 100) && row[8..].iter().all(|sample| *sample == 180)
    });
    luma && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == 90)
}

fn three_d_pcm() -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(25);
    w.align_zeros();
    for _row in 0..16 {
        for col in 0..16 {
            w.bytes.push(if col < 8 { 100 } else { 180 });
        }
    }
    w.bytes.extend(std::iter::repeat_n(90, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    let mut out = annex_b_nal(3, 15, &three_d_subset(false));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(3, 21, &[0x80, 0x42], &w.bytes));
    out
}

fn three_d_skip(weighted: bool) -> Vec<u8> {
    annex_b_extension(1, 21, &[0x80, 0x60], &three_d_skip_body(weighted))
}

fn three_d_skip_body(weighted: bool) -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 1, 1, true, false);
    w.bit(weighted);
    if !weighted {
        w.ue(1);
        w.trailing();
    }
    w.bytes
}

fn three_d_annex(header: &[u8; 2]) -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, &three_d_subset(false));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(
        3,
        21,
        header,
        &idr_slice(true, true, false),
    ));
    out
}

#[test]
fn a_depth_range_scales_the_predicted_luma() {
    let idr = three_d_pcm_ranges();
    let predicted = three_d_weighted();
    let mut decoder = open();
    let mut pictures = push_pictures(&mut decoder, &idr);
    pictures.extend(push_pictures(&mut decoder, &predicted));
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 2);
    assert!(split_depth(&pictures[0]));
    assert!(weighted_depth(&pictures[1]));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let mut annex = idr.clone();
        annex.extend(predicted);
        assert!(
            ffmpeg_bytes(&ffmpeg, &annex).is_empty(),
            "ffmpeg decoded a 3d view"
        );
    }
}

fn weighted_depth(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    let luma = y.chunks(16).all(|row| {
        row[..8].iter().all(|sample| *sample == 152) && row[8..].iter().all(|sample| *sample == 204)
    });
    luma && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == 90)
}

fn three_d_pcm_ranges() -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(25);
    w.align_zeros();
    for _row in 0..16 {
        for col in 0..16 {
            w.bytes.push(if col < 8 { 100 } else { 180 });
        }
    }
    w.bytes.extend(std::iter::repeat_n(90, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    let mut out = annex_b_nal(3, 15, &three_d_sequence(false, 31, 32));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(3, 21, &[0x80, 0x42], &w.bytes));
    out
}

fn three_d_weighted() -> Vec<u8> {
    let mut out = depth_parameter(31, 33);
    let mut w = Writer::new();
    ref_p_header(&mut w, 1, 1, true, false);
    w.bit(true);
    w.ue(1);
    w.ue(1);
    w.trailing();
    out.extend(annex_b_extension(1, 21, &[0x80, 0x60], &w.bytes));
    out
}

fn depth_parameter(near_exp: u32, far_exp: u32) -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(1);
    w.ue(2);
    w.ue(0);
    write_depth_ranges(&mut w, near_exp, far_exp);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.trailing();
    annex_b_nal(3, 16, &w.bytes)
}

fn write_depth_ranges(w: &mut Writer, near_exp: u32, far_exp: u32) {
    w.bit(true);
    w.bit(true);
    write_depth_value(w, near_exp);
    write_depth_value(w, far_exp);
}

fn write_depth_value(w: &mut Writer, exp: u32) {
    w.bits(0, 5);
    w.bit(false);
    w.bits(exp, 7);
    w.bits(0, 1);
}

/// Profile 139. The MVCD view is view id 0, depth and texture, and the 3D
/// tools are off unless `prediction` is set.
fn three_d_subset(prediction: bool) -> Vec<u8> {
    three_d_sequence(prediction, 0, 0)
}

fn three_d_sequence(prediction: bool, near_exp: u32, far_exp: u32) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(139, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.bits(30, 8);
    w.ue(0);
    w.bits(0, 3);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.ue(if near_exp == 0 { 0 } else { 2 });
    w.ue(0);
    if near_exp != 0 {
        write_depth_ranges(&mut w, near_exp, far_exp);
    }
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.bit(prediction);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

/// Two paired views. View 1's depth reference lists are empty, so the paired
/// flag copies that emptiness onto the texture interview lists. Four reference
/// frames keep both views' pictures.
fn three_d_two_views() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(139, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(4);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(1);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.ue(1);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.bits(30, 8);
    w.ue(0);
    w.bits(0, 3);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn depth_annex(header: &[u8; 3]) -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, &mvcd_subset(138, false));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(
        3,
        21,
        header,
        &idr_slice(true, true, false),
    ));
    out
}

fn mvcd_subset(profile: u8, vui: bool) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(u32::from(profile), 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.bits(30, 8);
    w.ue(0);
    w.bits(0, 3);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.ue(1);
    w.bit(vui);
    if !vui {
        w.bit(false);
        w.bit(false);
        w.trailing();
    }
    w.bytes
}

fn mvc_subset(profile: u8, vui: bool) -> Vec<u8> {
    mvc_views(profile, vui, false)
}

fn mvc_views(profile: u8, vui: bool, anchor_base: bool) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(u32::from(profile), 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(1);
    w.ue(0);
    w.ue(1);
    if anchor_base {
        w.ue(1);
        w.ue(0);
    } else {
        w.ue(0);
    }
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.bits(30, 8);
    w.ue(0);
    w.bits(0, 3);
    w.ue(0);
    w.ue(1);
    w.ue(1);
    w.bit(vui);
    if !vui {
        w.bit(false);
        w.trailing();
    }
    w.bytes
}

/// IDR, dependency 1, quality 0, inter-layer prediction on, output, reserved 3.
fn ilp_header() -> [u8; 3] {
    [0xc0, 0x10, 0x07]
}

fn pcm_annex() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &pcm_slice(200, 40)));
    out
}

fn inter_layer_annex() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 15, &layer_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &pcm_slice(200, 40)));
    out.extend(annex_b_extension(
        3,
        20,
        &ilp_header(),
        &base_mode_slice(false),
    ));
    out
}

fn residual_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 15, &uncropped_layer()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &residual_pcm()));
    out
}

fn residual_layer_packet() -> Vec<u8> {
    let mut out = motion_base_packet();
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &residual_enhancement(),
    ));
    out
}

fn residual_pcm() -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(25);
    w.align_zeros();
    for _row in 0..16 {
        for col in 0..16 {
            w.bytes.push(if col < 8 { 100 } else { 180 });
        }
    }
    w.bytes.extend(std::iter::repeat_n(90, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    w.bytes
}

fn moving_base_packet() -> Vec<u8> {
    annex_b_nal(0, 1, &moving_base_slice())
}

fn moving_residual_packet() -> Vec<u8> {
    let mut out = moving_base_packet();
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &still_enhancement(),
    ));
    out
}

fn moving_base_slice() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(0);
    w.se(32);
    w.se(0);
    w.ue(2);
    w.se(0);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn still_enhancement() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.se(0);
    w.se(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn residual_enhancement() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.se(32);
    w.se(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn split_luma(picture: &VideoOutput, left: u8, right: u8) -> bool {
    let y = picture.picture.y();
    (0..16).all(|row| {
        y[row * 16..row * 16 + 8]
            .iter()
            .all(|sample| *sample == left)
            && y[row * 16 + 8..row * 16 + 16]
                .iter()
                .all(|sample| *sample == right)
    }) && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == 90)
}

fn base_residual_picture(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    (0..16).all(|row| {
        (0..16).all(|col| {
            let sample = y[row * 16 + col];
            let want = if row < 4 && col < 4 {
                103
            } else if col < 8 {
                100
            } else {
                180
            };
            sample == want
        })
    }) && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == 90)
}

fn shifted_residual_picture(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    (0..16).all(|row| {
        (0..16).all(|col| {
            let want = if row < 4 && col < 4 { 183 } else { 180 };
            y[row * 16 + col] == want
        })
    }) && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == 90)
}

fn older_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &two_ref_sps());
    out.extend(annex_b_nal(3, 15, &uncropped_layer()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &residual_pcm()));
    out
}

fn recent_reference() -> Vec<u8> {
    annex_b_nal(3, 1, &recent_slice())
}

fn future_reference() -> Vec<u8> {
    annex_b_nal(3, 1, &future_slice())
}

fn list1_base_packet() -> Vec<u8> {
    annex_b_nal(0, 1, &list1_base_slice())
}

fn list1_inherit_packet() -> Vec<u8> {
    let mut out = list1_base_packet();
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &list1_inherit(),
    ));
    out
}

fn list1_residual_packet() -> Vec<u8> {
    let mut out = list1_base_packet();
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &list1_residual(),
    ));
    out
}

fn older_base_packet() -> Vec<u8> {
    annex_b_nal(0, 1, &older_base_slice())
}

fn older_inherit_packet() -> Vec<u8> {
    let mut out = older_base_packet();
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &older_inherit(),
    ));
    out
}

fn older_residual_packet() -> Vec<u8> {
    let mut out = older_base_packet();
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &older_residual(),
    ));
    out
}

fn two_ref_sps() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(66, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(2);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn ref_p_header(w: &mut Writer, frame_num: u32, poc: u32, referenced: bool, two_refs: bool) {
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.bits(frame_num, 4);
    w.bits(poc, 4);
    w.bit(two_refs);
    if two_refs {
        w.ue(1);
    }
    w.bit(false);
    if referenced {
        w.bit(false);
    }
    w.se(0);
    w.ue(1);
}

fn future_slice() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 1, 2, true, false);
    w.ue(0);
    w.ue(30);
    w.align_zeros();
    w.bytes.extend(std::iter::repeat_n(40, 256));
    w.bytes.extend(std::iter::repeat_n(90, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    w.bytes
}

fn b_header(w: &mut Writer) {
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.bits(2, 4);
    w.bits(1, 4);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.se(0);
    w.ue(1);
}

fn list1_base_slice() -> Vec<u8> {
    let mut w = Writer::new();
    b_header(&mut w);
    w.ue(0);
    w.ue(2);
    w.se(0);
    w.se(0);
    w.ue(2);
    w.se(0);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn list1_inherit() -> Vec<u8> {
    let mut w = Writer::new();
    b_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn bipred_base_packet() -> Vec<u8> {
    annex_b_nal(0, 1, &bipred_base_slice())
}

fn bipred_inherit_packet() -> Vec<u8> {
    let mut out = bipred_base_packet();
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &bipred_inherit(),
    ));
    out
}

fn bipred_residual_packet() -> Vec<u8> {
    let mut out = bipred_base_packet();
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &bipred_residual(),
    ));
    out
}

fn bipred_base_slice() -> Vec<u8> {
    let mut w = Writer::new();
    b_header(&mut w);
    w.ue(0);
    w.ue(3);
    w.se(0);
    w.se(0);
    w.se(0);
    w.se(0);
    w.ue(2);
    w.se(0);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn bipred_inherit() -> Vec<u8> {
    let mut w = Writer::new();
    b_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn bipred_residual() -> Vec<u8> {
    let mut w = Writer::new();
    b_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(1);
    w.se(0);
    w.se(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn corner_on_split(picture: &VideoOutput, corner: u8, left: u8, right: u8, chroma: u8) -> bool {
    let y = picture.picture.y();
    (0..16).all(|row| {
        (0..16).all(|col| {
            let want = if row < 4 && col < 4 {
                corner
            } else if col < 8 {
                left
            } else {
                right
            };
            y[row * 16 + col] == want
        })
    }) && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == chroma)
}

fn list1_residual() -> Vec<u8> {
    let mut w = Writer::new();
    b_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(1);
    w.se(0);
    w.se(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn recent_slice() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 1, 1, true, false);
    w.ue(0);
    w.ue(30);
    w.align_zeros();
    w.bytes.extend(std::iter::repeat_n(40, 256));
    w.bytes.extend(std::iter::repeat_n(90, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    w.bytes
}

fn older_base_slice() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 2, 2, false, true);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.se(0);
    w.se(0);
    w.ue(2);
    w.se(0);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn older_inherit() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 2, 2, false, true);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn older_residual() -> Vec<u8> {
    let mut w = Writer::new();
    ref_p_header(&mut w, 2, 2, false, false);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.se(0);
    w.se(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn corner_on_flat(picture: &VideoOutput, corner: u8, flat: u8, chroma: u8) -> bool {
    let y = picture.picture.y();
    (0..16).all(|row| {
        (0..16).all(|col| {
            let want = if row < 4 && col < 4 { corner } else { flat };
            y[row * 16 + col] == want
        })
    }) && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == chroma)
}

fn motion_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 15, &uncropped_layer()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &pcm_slice(150, 90)));
    out
}

fn motion_base_packet() -> Vec<u8> {
    annex_b_nal(0, 1, &motion_p_slice())
}

fn motion_layer_packet() -> Vec<u8> {
    let mut out = motion_base_packet();
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &motion_enhancement(),
    ));
    out
}

fn motion_p_slice() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(0);
    w.se(0);
    w.se(0);
    w.ue(2);
    w.se(0);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn both_tools_packet() -> Vec<u8> {
    let mut out = moving_base_packet();
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &both_tools_enhancement(),
    ));
    out
}

fn both_tools_enhancement() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(2);
    w.se(0);
    w.bit(true);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn both_tools_picture(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    (0..16).all(|row| {
        (0..16).all(|col| {
            let want = if row < 4 && col < 8 { 183 } else { 180 };
            y[row * 16 + col] == want
        })
    }) && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == 90)
}

fn motion_enhancement() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn motion_p_header(w: &mut Writer) {
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.bits(1, 4);
    w.bits(1, 4);
    w.bit(false);
    w.bit(false);
    w.se(0);
    w.ue(1);
}

fn planes_of(picture: &VideoOutput) -> Vec<u8> {
    let mut raw = picture.picture.y().to_vec();
    raw.extend_from_slice(picture.picture.u());
    raw.extend_from_slice(picture.picture.v());
    raw
}

fn residual_annex() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 15, &uncropped_layer()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &pcm_slice(200, 40)));
    out.extend(annex_b_extension(
        3,
        20,
        &ilp_header(),
        &base_mode_residual(),
    ));
    out
}

fn residual_layer() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 15, &layer_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &pcm_slice(200, 40)));
    out.extend(annex_b_extension(
        3,
        20,
        &ilp_header(),
        &base_mode_slice(true),
    ));
    out
}

fn base_with_broken_layer() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 15, &layer_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &idr_slice(true, true, false)));
    out.extend(annex_b_extension(3, 20, &ilp_header(), &[0x80]));
    out
}

fn layer_subset() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(83, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.ue(4);
    w.ue(0);
    w.ue(4);
    w.bit(false);
    w.bit(true);
    w.bits(0, 2);
    w.bit(false);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn pcm_slice(luma: u8, chroma: u8) -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(25);
    w.align_zeros();
    w.bytes.extend(std::iter::repeat_n(luma, 256));
    w.bytes.extend(std::iter::repeat_n(chroma, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    w.bytes
}

fn uncropped_layer() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(83, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bits(0, 2);
    w.bit(false);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn base_mode_residual() -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    // Intra coded-block pattern 1: the top-left 8×8. QP delta 0.
    w.ue(29);
    w.se(0);
    // Trailing +1 at DC of the first 4×4, then three empty blocks.
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn base_mode_slice(residual: bool) -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(residual);
    w.ue(3);
    w.trailing();
    w.bytes
}

fn i_header(w: &mut Writer) {
    w.ue(0);
    w.ue(2);
    w.ue(0);
    w.bits(0, 4);
    w.ue(0);
    w.bits(0, 4);
    w.bit(false);
    w.bit(false);
    w.se(0);
    w.ue(1);
}

fn open() -> H264Decoder {
    let mut params = CodecParams::video(CodecId::H264, 16, 16);
    params.nal_length_size = 0;
    H264Decoder::open(&params).expect("open")
}

fn decode(annex: &[u8]) -> VideoOutput {
    decode_with(&mut open(), annex)
}

fn decode_with(decoder: &mut H264Decoder, annex: &[u8]) -> VideoOutput {
    let mut pictures = decoder
        .push(&Packet::video(annex.to_vec(), 0, 1, 30, true))
        .expect("decode");
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 1);
    pictures.remove(0)
}

fn flat(picture: &VideoOutput) -> bool {
    painted(picture, 128, 128)
}

fn painted(picture: &VideoOutput, luma: u8, chroma: u8) -> bool {
    picture.picture.y().iter().all(|sample| *sample == luma)
        && picture
            .picture
            .u()
            .iter()
            .chain(picture.picture.v())
            .all(|sample| *sample == chroma)
}

fn paint(edge: usize, luma: u8, chroma: u8) -> Vec<u8> {
    let mut raw = vec![luma; edge * edge];
    raw.extend(std::iter::repeat_n(chroma, (edge / 2) * (edge / 2) * 2));
    raw
}

fn wide_ref_sps() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(66, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(4);
    w.bit(false);
    w.ue(1);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn four_ref_sps() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(66, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(4);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn baseline_sps() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(66, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn baseline_subset() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(66, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn truncated_scalable() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(83, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn subset(profile: u8, restriction: bool, spatial: u32, vui: bool) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(u32::from(profile), 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bits(spatial, 2);
    if spatial <= 2 {
        w.bit(false);
        w.bits(1, 2);
        if spatial == 1 {
            w.bit(false);
            w.bits(1, 2);
            w.se(0);
            w.se(0);
            w.se(0);
            w.se(0);
        }
        w.bit(false);
        w.bit(restriction);
        w.bit(vui);
        if !vui {
            w.bit(false);
            w.trailing();
        }
    }
    w.bytes
}

fn intra8_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &high_sps());
    out.extend(annex_b_nal(3, 15, &high_subset()));
    out.extend(annex_b_nal(3, 8, &pps_8x8()));
    out.extend(annex_b_nal(3, 5, &pcm_slice(200, 40)));
    out
}

fn intra8_layer() -> Vec<u8> {
    let mut out = intra8_idr();
    out.extend(annex_b_extension(3, 20, &ilp_header(), &intra8_slice()));
    out
}

fn intra8_slice() -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.ue(29);
    w.bit(true);
    w.se(0);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn eight_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &high_sps());
    out.extend(annex_b_nal(3, 15, &high_subset()));
    out.extend(annex_b_nal(3, 8, &pps_8x8()));
    out.extend(annex_b_nal(3, 5, &pcm_slice(150, 90)));
    out
}

fn eight_layer() -> Vec<u8> {
    let mut out = annex_b_nal(0, 1, &eight_base_slice());
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &eight_enhancement(),
    ));
    out
}

fn eight_p_slice() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(0);
    w.se(0);
    w.se(0);
    eight_residual(&mut w);
    w.trailing();
    w.bytes
}

fn eight_base_slice() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(0);
    w.se(0);
    w.se(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn eight_enhancement() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    eight_residual(&mut w);
    w.trailing();
    w.bytes
}

fn eight_residual(w: &mut Writer) {
    w.ue(2);
    w.bit(true);
    w.se(0);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
}

fn adaptive_annex() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &high_sps_width(1));
    out.extend(annex_b_nal(3, 15, &high_subset_width(1)));
    out.extend(annex_b_nal(3, 8, &pps_8x8()));
    out.extend(annex_b_nal(3, 5, &wide_pcm()));
    out.extend(annex_b_extension(3, 20, &ilp_header(), &adaptive_slice()));
    out
}

fn wide_pcm() -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    for luma in [100u8, 180] {
        w.ue(25);
        w.align_zeros();
        w.bytes.extend(std::iter::repeat_n(luma, 256));
        w.bytes.extend(std::iter::repeat_n(90, 128));
        w.bit = w.bytes.len() * 8;
    }
    w.trailing();
    w.bytes
}

fn adaptive_slice() -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(3);
    w.bit(false);
    w.ue(25);
    w.align_zeros();
    w.bytes.extend(std::iter::repeat_n(40, 256));
    w.bytes.extend(std::iter::repeat_n(40, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    w.bytes
}

fn residual_flag_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &high_sps_width(1));
    out.extend(annex_b_nal(3, 15, &high_subset_width(1)));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &wide_flat_pcm(150, 90)));
    out
}

fn residual_flag_base() -> Vec<u8> {
    annex_b_nal(0, 1, &residual_flag_base_slice())
}

fn residual_flag_layer() -> Vec<u8> {
    let mut out = residual_flag_base();
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &residual_flag_enhancement(),
    ));
    out
}

fn wide_flat_pcm(luma: u8, chroma: u8) -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    for _ in 0..2 {
        w.ue(25);
        w.align_zeros();
        w.bytes.extend(std::iter::repeat_n(luma, 256));
        w.bytes.extend(std::iter::repeat_n(chroma, 128));
        w.bit = w.bytes.len() * 8;
    }
    w.trailing();
    w.bytes
}

fn residual_flag_base_slice() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    for _ in 0..2 {
        w.ue(0);
        w.ue(0);
        w.se(0);
        w.se(0);
        w.ue(2);
        w.se(0);
        w.bits(1, 2);
        w.bit(false);
        w.bit(true);
        w.bit(true);
        w.bit(true);
        w.bit(true);
    }
    w.trailing();
    w.bytes
}

fn residual_flag_enhancement() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn residual_corners(picture: &VideoOutput, left: u8, right: u8, fill: u8) -> bool {
    let y = picture.picture.y();
    (0..16).all(|row| {
        (0..32).all(|col| {
            let corner = if col < 16 { left } else { right };
            let want = if row < 4 && (col < 4 || (16..20).contains(&col)) {
                corner
            } else {
                fill
            };
            y[row * 32 + col] == want
        })
    }) && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == 90)
}

fn motion_flag_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &high_sps_width(1));
    out.extend(annex_b_nal(3, 15, &high_subset_width(1)));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &striped_pcm(100, 180, 90)));
    out
}

fn motion_flag_base() -> Vec<u8> {
    annex_b_nal(0, 1, &motion_flag_base_slice())
}

fn motion_flag_layer() -> Vec<u8> {
    let mut out = motion_flag_base();
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &motion_flag_enhancement(),
    ));
    out
}

fn striped_pcm(left: u8, right: u8, chroma: u8) -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    for _ in 0..2 {
        w.ue(25);
        w.align_zeros();
        for _ in 0..16 {
            w.bytes.extend(std::iter::repeat_n(left, 8));
            w.bytes.extend(std::iter::repeat_n(right, 8));
        }
        w.bytes.extend(std::iter::repeat_n(chroma, 128));
        w.bit = w.bytes.len() * 8;
    }
    w.trailing();
    w.bytes
}

fn motion_flag_base_slice() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(0);
    w.se(32);
    w.se(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.se(0);
    w.se(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn motion_flag_enhancement() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.se(0);
    w.se(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.se(0);
    w.se(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn striped(picture: &VideoOutput, left: u8, right: u8) -> bool {
    let y = picture.picture.y();
    (0..16).all(|row| {
        (0..32).all(|col| {
            let band = col % 16;
            let want = if band < 8 { left } else { right };
            y[row * 32 + col] == want
        })
    })
}

fn shifted_pair(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    (0..16).all(|row| {
        (0..32).all(|col| {
            let want = if !(8..16).contains(&col) { 180 } else { 100 };
            y[row * 32 + col] == want
        })
    }) && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == 90)
}

fn motion_flag_picture(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    (0..16).all(|row| {
        (0..32).all(|col| {
            let want = if col < 8 { 100 } else { 180 };
            y[row * 32 + col] == want
        })
    }) && picture
        .picture
        .u()
        .iter()
        .chain(picture.picture.v())
        .all(|sample| *sample == 90)
}

fn paint_row(left: u8, right: u8) -> Vec<u8> {
    let mut row = vec![left; 16];
    row.extend(std::iter::repeat_n(right, 16));
    row
}

fn spatial_subset() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(86, 8);
    high_body_size(&mut w, 1, 1);
    w.bit(true);
    w.bits(1, 2);
    w.bit(true);
    w.bits(1, 2);
    w.bit(true);
    w.bits(1, 2);
    w.se(0);
    w.se(0);
    w.se(0);
    w.se(0);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn resample_base() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &high_sps());
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &resample_pcm()));
    out
}

fn resample_annex() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &high_sps());
    out.extend(annex_b_nal(3, 15, &spatial_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &resample_pcm()));
    out.extend(annex_b_extension(3, 20, &ilp_header(), &resample_slice()));
    out
}

fn resample_pcm() -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(25);
    w.align_zeros();
    for _ in 0..16 {
        w.bytes.extend(std::iter::repeat_n(40, 8));
        w.bytes.extend(std::iter::repeat_n(200, 8));
    }
    for _ in 0..2 {
        for _ in 0..8 {
            w.bytes.extend(std::iter::repeat_n(40, 4));
            w.bytes.extend(std::iter::repeat_n(200, 4));
        }
    }
    w.bit = w.bytes.len() * 8;
    w.trailing();
    w.bytes
}

fn resample_slice() -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.ue(29);
    w.se(0);
    w.bits(1, 2);
    w.bit(false);
    for _ in 0..4 {
        w.bit(true);
    }
    for _ in 0..3 {
        w.ue(3);
    }
    w.trailing();
    w.bytes
}

fn resample_base_picture(picture: &VideoOutput) -> bool {
    let y = picture.picture.y();
    let chroma = |plane: &[u8]| {
        (0..8).all(|row| {
            plane[row * 8..row * 8 + 4]
                .iter()
                .all(|sample| *sample == 40)
                && plane[row * 8 + 4..row * 8 + 8]
                    .iter()
                    .all(|sample| *sample == 200)
        })
    };
    (0..16).all(|row| {
        y[row * 16..row * 16 + 8].iter().all(|sample| *sample == 40)
            && y[row * 16 + 8..row * 16 + 16]
                .iter()
                .all(|sample| *sample == 200)
    }) && chroma(picture.picture.u())
        && chroma(picture.picture.v())
}

fn crop_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &high_sps());
    out.extend(annex_b_nal(3, 15, &cropped_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &resample_pcm()));
    out.extend(annex_b_extension(3, 20, &ilp_header(), &crop_intra()));
    out
}

fn cropped_subset() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(86, 8);
    high_body_size(&mut w, 2, 0);
    w.bit(true);
    w.bits(1, 2);
    w.bit(true);
    w.bits(1, 2);
    w.bit(true);
    w.bits(1, 2);
    w.se(0);
    w.se(0);
    w.se(8);
    w.se(0);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn crop_intra() -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    for _ in 0..3 {
        w.ue(3);
    }
    w.trailing();
    w.bytes
}

fn crop_p() -> Vec<u8> {
    let mut out = annex_b_nal(0, 1, &motion_scale_base());
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &crop_enhancement(),
    ));
    out
}

fn crop_enhancement() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    for _ in 0..2 {
        w.ue(0);
        w.bit(true);
        w.ue(0);
    }
    w.ue(0);
    w.bit(false);
    w.ue(30);
    w.align_zeros();
    w.bytes.extend(std::iter::repeat_n(50, 256));
    w.bytes.extend(std::iter::repeat_n(50, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    w.bytes
}

fn cropped_reference(picture: &VideoOutput) -> bool {
    let luma = upsampled_luma();
    let chroma = upsampled_chroma();
    let y = picture.picture.y();
    let luma_ok = (0..16).all(|row| {
        y[row * 48..row * 48 + 32] == luma
            && y[row * 48 + 32..row * 48 + 48]
                .iter()
                .all(|sample| *sample == 128)
    });
    let chroma_ok = |plane: &[u8]| {
        (0..8).all(|row| {
            plane[row * 24..row * 24 + 16] == chroma
                && plane[row * 24 + 16..row * 24 + 24]
                    .iter()
                    .all(|sample| *sample == 128)
        })
    };
    luma_ok && chroma_ok(picture.picture.u()) && chroma_ok(picture.picture.v())
}

fn crop_change_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &high_sps());
    out.extend(annex_b_nal(3, 15, &crop_change_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &resample_pcm()));
    out.extend(annex_b_extension(
        3,
        20,
        &ilp_header(),
        &motion_scale_intra(),
    ));
    out
}

fn crop_change_subset() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(86, 8);
    high_body_size(&mut w, 1, 1);
    w.bit(true);
    w.bits(1, 2);
    w.bit(true);
    w.bits(1, 2);
    w.bit(true);
    w.bits(1, 2);
    w.se(2);
    w.se(0);
    w.se(-2);
    w.se(0);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn crop_change_p() -> Vec<u8> {
    let mut out = annex_b_nal(0, 1, &still_base());
    out.extend(annex_b_nal(3, 15, &spatial_subset()));
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &motion_scale_enhancement(),
    ));
    out
}

fn still_base() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(0);
    w.se(0);
    w.se(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn crop_change_reference(picture: &VideoOutput) -> bool {
    let luma = upsampled_luma();
    let chroma = upsampled_chroma();
    let y = picture.picture.y();
    let luma_ok = (0..32).all(|row| {
        y[row * 32..row * 32 + 4]
            .iter()
            .all(|sample| *sample == 128)
            && y[row * 32 + 4..row * 32 + 32] == luma[..28]
    });
    let chroma_ok = |plane: &[u8]| {
        (0..16).all(|row| {
            plane[row * 16..row * 16 + 2]
                .iter()
                .all(|sample| *sample == 128)
                && plane[row * 16 + 2..row * 16 + 16] == chroma[..14]
        })
    };
    luma_ok && chroma_ok(picture.picture.u()) && chroma_ok(picture.picture.v())
}

fn slice_crop(w: &mut Writer) {
    w.bit(true);
    w.bits(1, 2);
    w.se(8);
    w.se(0);
    w.se(-8);
    w.se(0);
}

fn slice_crop_subset() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(86, 8);
    high_body_size(&mut w, 1, 1);
    w.bit(true);
    w.bits(2, 2);
    w.bit(true);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn slice_crop_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &high_sps());
    out.extend(annex_b_nal(3, 15, &slice_crop_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &resample_pcm()));
    out.extend(annex_b_extension(3, 20, &ilp_header(), &slice_crop_intra()));
    out
}

fn slice_crop_intra() -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    slice_crop(&mut w);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    for _ in 0..4 {
        w.ue(3);
    }
    w.trailing();
    w.bytes
}

fn slice_crop_p() -> Vec<u8> {
    let mut out = annex_b_nal(0, 1, &residual_scale_base());
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &slice_crop_enhancement(),
    ));
    out
}

fn slice_crop_enhancement() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    slice_crop(&mut w);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(1);
    w.bit(true);
    w.ue(0);
    w.ue(1);
    w.bit(true);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn shifted_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &high_sps());
    out.extend(annex_b_nal(3, 15, &shifted_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &resample_pcm()));
    out.extend(annex_b_extension(
        3,
        20,
        &ilp_header(),
        &motion_scale_intra(),
    ));
    out
}

fn shifted_subset() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(86, 8);
    high_body_size(&mut w, 1, 1);
    w.bit(true);
    w.bits(1, 2);
    w.bit(true);
    w.bits(1, 2);
    w.bit(true);
    w.bits(1, 2);
    w.se(8);
    w.se(0);
    w.se(-8);
    w.se(0);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn shifted_residual_p() -> Vec<u8> {
    let mut out = annex_b_nal(0, 1, &residual_scale_base());
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &shifted_enhancement(),
    ));
    out
}

fn shifted_enhancement() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(1);
    w.bit(true);
    w.ue(0);
    w.ue(1);
    w.bit(true);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn shifted_reference(picture: &VideoOutput) -> bool {
    let luma = upsampled_luma();
    let chroma = upsampled_chroma();
    let y = picture.picture.y();
    let luma_ok = (0..32).all(|row| {
        y[row * 32..row * 32 + 16]
            .iter()
            .all(|sample| *sample == 128)
            && y[row * 32 + 16..row * 32 + 32] == luma[..16]
    });
    let chroma_ok = |plane: &[u8]| {
        (0..16).all(|row| {
            plane[row * 16..row * 16 + 8]
                .iter()
                .all(|sample| *sample == 128)
                && plane[row * 16 + 8..row * 16 + 16] == chroma[..8]
        })
    };
    luma_ok && chroma_ok(picture.picture.u()) && chroma_ok(picture.picture.v())
}

fn shifted_residual(picture: &VideoOutput) -> bool {
    let luma = upsampled_luma();
    let y = picture.picture.y();
    let luma_ok = (0..32).all(|row| {
        (0..32).all(|col| {
            let mut want = if col < 16 { 128 } else { luma[col - 16] };
            if row < 8 && (16..24).contains(&col) {
                want = 43;
            }
            y[row * 32 + col] == want
        })
    });
    shifted_reference_chroma(picture) && luma_ok
}

fn shifted_reference_chroma(picture: &VideoOutput) -> bool {
    let chroma = upsampled_chroma();
    let chroma_ok = |plane: &[u8]| {
        (0..16).all(|row| {
            plane[row * 16..row * 16 + 8]
                .iter()
                .all(|sample| *sample == 128)
                && plane[row * 16 + 8..row * 16 + 16] == chroma[..8]
        })
    };
    chroma_ok(picture.picture.u()) && chroma_ok(picture.picture.v())
}

fn crop_residual_p() -> Vec<u8> {
    let mut out = annex_b_nal(0, 1, &residual_scale_base());
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &crop_residual_enhancement(),
    ));
    out
}

fn crop_residual_enhancement() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    for _ in 0..2 {
        w.ue(0);
        w.bit(true);
        w.ue(0);
    }
    w.ue(1);
    w.trailing();
    w.bytes
}

fn cropped_residual(picture: &VideoOutput) -> bool {
    let luma = upsampled_luma();
    let chroma = upsampled_chroma();
    let y = picture.picture.y();
    let luma_ok = (0..16).all(|row| {
        (0..48).all(|col| {
            let mut want = if col < 32 { luma[col] } else { 128 };
            if row < 4 && col < 8 {
                want = 43;
            }
            y[row * 48 + col] == want
        })
    });
    let chroma_ok = |plane: &[u8]| {
        (0..8).all(|row| {
            plane[row * 24..row * 24 + 16] == chroma
                && plane[row * 24 + 16..row * 24 + 24]
                    .iter()
                    .all(|sample| *sample == 128)
        })
    };
    luma_ok && chroma_ok(picture.picture.u()) && chroma_ok(picture.picture.v())
}

fn cropped_motion(picture: &VideoOutput) -> bool {
    let luma = [
        165, 215, 205, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    ];
    let y = picture.picture.y();
    let luma_ok = (0..16).all(|row| {
        y[row * 48..row * 48 + 16] == luma
            && y[row * 48 + 16..row * 48 + 32]
                .iter()
                .all(|sample| *sample == 128)
            && y[row * 48 + 32..row * 48 + 48]
                .iter()
                .all(|sample| *sample == 50)
    });
    let chroma_ok = |plane: &[u8]| {
        (0..8).all(|row| {
            plane[row * 24] == 160
                && plane[row * 24 + 1..row * 24 + 8]
                    .iter()
                    .all(|sample| *sample == 200)
                && plane[row * 24 + 8..row * 24 + 16]
                    .iter()
                    .all(|sample| *sample == 128)
                && plane[row * 24 + 16..row * 24 + 24]
                    .iter()
                    .all(|sample| *sample == 50)
        })
    };
    luma_ok && chroma_ok(picture.picture.u()) && chroma_ok(picture.picture.v())
}

fn motion_scale_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &high_sps());
    out.extend(annex_b_nal(3, 15, &spatial_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &resample_pcm()));
    out.extend(annex_b_extension(
        3,
        20,
        &ilp_header(),
        &motion_scale_intra(),
    ));
    out
}

fn motion_scale_intra() -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    for _ in 0..4 {
        w.ue(3);
    }
    w.trailing();
    w.bytes
}

fn motion_scale_p() -> Vec<u8> {
    let mut out = annex_b_nal(0, 1, &motion_scale_base());
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &motion_scale_enhancement(),
    ));
    out
}

fn motion_scale_base() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(0);
    w.se(32);
    w.se(0);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn residual_scale_p() -> Vec<u8> {
    let mut out = annex_b_nal(0, 1, &residual_scale_base());
    out.extend(annex_b_extension(
        0,
        20,
        &[0x80, 0x10, 0x07],
        &residual_scale_enhancement(),
    ));
    out
}

fn residual_scale_base() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(0);
    w.se(0);
    w.se(0);
    w.ue(2);
    w.se(0);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn residual_scale_enhancement() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(true);
    for _ in 0..4 {
        w.ue(0);
        w.ue(0);
    }
    w.trailing();
    w.bytes
}

fn residual_resampled(picture: &VideoOutput) -> bool {
    let luma = upsampled_luma();
    let chroma = upsampled_chroma();
    let y = picture.picture.y();
    let luma_ok = (0..32).all(|row| {
        (0..32).all(|col| {
            let mut want = luma[col];
            if row < 8 && col < 8 {
                want = 43;
            }
            y[row * 32 + col] == want
        })
    });
    let chroma_ok = |plane: &[u8]| (0..16).all(|row| plane[row * 16..row * 16 + 16] == chroma);
    luma_ok && chroma_ok(picture.picture.u()) && chroma_ok(picture.picture.v())
}

fn motion_scale_enhancement() -> Vec<u8> {
    let mut w = Writer::new();
    motion_p_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    for _ in 0..4 {
        w.ue(0);
        w.ue(0);
    }
    w.trailing();
    w.bytes
}

fn upsampled_reference(picture: &VideoOutput) -> bool {
    let luma = upsampled_luma();
    let chroma = upsampled_chroma();
    let y = picture.picture.y();
    (0..32).all(|row| y[row * 32..row * 32 + 32] == luma)
        && (0..16).all(|row| picture.picture.u()[row * 16..row * 16 + 16] == chroma)
        && (0..16).all(|row| picture.picture.v()[row * 16..row * 16 + 16] == chroma)
}

fn scaled_motion_picture(picture: &VideoOutput) -> bool {
    let luma = [
        165, 215, 205, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    ];
    let chroma = [
        160, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    ];
    let y = picture.picture.y();
    (0..32).all(|row| y[row * 32..row * 32 + 32] == luma)
        && (0..16).all(|row| picture.picture.u()[row * 16..row * 16 + 16] == chroma)
        && (0..16).all(|row| picture.picture.v()[row * 16..row * 16 + 16] == chroma)
}

fn upsampled_luma() -> [u8; 32] {
    [
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 35, 25, 75, 165, 215, 205, 200, 200,
        200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    ]
}

fn upsampled_chroma() -> [u8; 16] {
    [
        40, 40, 40, 40, 40, 40, 40, 80, 160, 200, 200, 200, 200, 200, 200, 200,
    ]
}

fn resampled(picture: &VideoOutput) -> bool {
    let luma = [
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 35, 25, 75, 165, 215, 205, 200, 200,
        200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    ];
    let chroma = [
        40, 40, 40, 40, 40, 40, 40, 80, 160, 200, 200, 200, 200, 200, 200, 200,
    ];
    let y = picture.picture.y();
    let luma_ok = (0..32).all(|row| {
        (0..32).all(|col| {
            let mut want = luma[col];
            if row < 4 && col < 4 {
                want = 43;
            }
            y[row * 32 + col] == want
        })
    });
    let chroma_ok = |plane: &[u8]| (0..16).all(|row| plane[row * 16..row * 16 + 16] == chroma);
    luma_ok && chroma_ok(picture.picture.u()) && chroma_ok(picture.picture.v())
}

fn high_sps() -> Vec<u8> {
    high_sps_width(0)
}

fn high_sps_width(width_minus1: u32) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(100, 8);
    high_body(&mut w, width_minus1);
    w.trailing();
    w.bytes
}

fn high_subset() -> Vec<u8> {
    high_subset_width(0)
}

fn high_subset_width(width_minus1: u32) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(86, 8);
    high_body(&mut w, width_minus1);
    w.bit(true);
    w.bits(0, 2);
    w.bit(false);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn high_body(w: &mut Writer, width_minus1: u32) {
    high_body_size(w, width_minus1, 0);
}

fn high_body_size(w: &mut Writer, width_minus1: u32, height_minus1: u32) {
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(width_minus1);
    w.ue(height_minus1);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
}

fn pps_8x8() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bits(0, 2);
    w.se(0);
    w.se(0);
    w.se(0);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn pps() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bits(0, 2);
    w.se(0);
    w.se(0);
    w.se(0);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn idr_slice(base_quality: bool, restriction: bool, store_ref_base: bool) -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(2);
    w.ue(0);
    w.bits(0, 4);
    w.ue(0);
    w.bits(0, 4);
    if base_quality {
        w.bit(false);
        w.bit(false);
        if !restriction {
            w.bit(store_ref_base);
        }
    }
    w.se(0);
    w.ue(1);
    if !restriction {
        w.bits(0, 4);
        w.bits(15, 4);
    }
    w.ue(3);
    w.ue(0);
    w.se(0);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn ffmpeg_bin() -> Option<PathBuf> {
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"] {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn ffmpeg_bytes(ffmpeg: &std::path::Path, annex: &[u8]) -> Vec<u8> {
    spawn(ffmpeg, annex).stdout
}

fn ffmpeg_raw(ffmpeg: &std::path::Path, annex: &[u8]) -> Vec<u8> {
    let output = spawn(ffmpeg, annex);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn spawn(ffmpeg: &std::path::Path, annex: &[u8]) -> std::process::Output {
    let mut child = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-threads",
            "1",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("ffmpeg");
    std::io::Write::write_all(child.stdin.as_mut().expect("stdin"), annex).expect("write");
    child.wait_with_output().expect("wait")
}

struct Writer {
    bytes: Vec<u8>,
    bit: usize,
}

impl Writer {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit: 0,
        }
    }

    fn bit(&mut self, value: bool) {
        let index = self.bit / 8;
        if index == self.bytes.len() {
            self.bytes.push(0);
        }
        if value {
            self.bytes[index] |= 1 << (7 - (self.bit % 8));
        }
        self.bit += 1;
    }

    fn bits(&mut self, value: u32, n: u32) {
        for shift in (0..n).rev() {
            self.bit(((value >> shift) & 1) == 1);
        }
    }

    fn ue(&mut self, value: u32) {
        let code = value + 1;
        let zeros = 31 - code.leading_zeros();
        for _ in 0..zeros {
            self.bit(false);
        }
        self.bits(code, zeros + 1);
    }

    fn align_zeros(&mut self) {
        while !self.bit.is_multiple_of(8) {
            self.bit(false);
        }
    }

    fn se(&mut self, value: i32) {
        let code = if value > 0 {
            value.unsigned_abs() * 2 - 1
        } else {
            value.unsigned_abs() * 2
        };
        self.ue(code);
    }

    fn trailing(&mut self) {
        self.bit(true);
        while !self.bit.is_multiple_of(8) {
            self.bit(false);
        }
    }
}
