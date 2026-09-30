"""Integration tests for the cutlass v2 Python API.

Run after `maturin develop --features extension-module`:

    python -m pytest tests/ -q

The media tests bootstrap their own sample file by exporting a tiny project
to mp4, so no fixtures need to be checked in. Render/export tests require a
working GPU + encoder (they run on macOS dev machines).
"""

from __future__ import annotations

import struct
import zlib

import pytest

import cutlass
from cutlass import (
    Arrow,
    CutlassError,
    Ellipse,
    Heart,
    Line,
    MediaError,
    OverlapError,
    Polygon,
    Project,
    Rect,
    ShapeStroke,
    Solid,
    Star,
    Text,
    TextBackground,
    TextShadow,
    TextStroke,
    TrackKindError,
)

# --- fixtures ----------------------------------------------------------------


def write_solid_png(
    path: str, width: int, height: int, rgba: tuple[int, int, int, int]
) -> None:
    """Write a minimal RGBA PNG with a single solid color (stdlib only)."""
    r, g, b, a = rgba

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    row = bytes([r, g, b, a] * width)
    raw = b"".join(b"\x00" + row for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


@pytest.fixture
def p() -> Project:
    return Project("test", fps=30, canvas="16:9", background=(10, 20, 30))


@pytest.fixture(scope="session")
def media_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Export a 2s solid-color timeline to mp4 and hand back the path."""
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    src = Project("sample", fps=30, canvas="16:9", background=(0, 0, 0))
    src.add_track("sticker").add(Solid("#3060c0"), start=0.0, duration=2.0)
    frames = src.export(str(out))
    assert frames == 60
    return str(out)


@pytest.fixture(scope="session")
def image_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Write a small solid-color PNG for still-image import tests."""
    out = tmp_path_factory.mktemp("media") / "frame.png"
    write_solid_png(str(out), 64, 48, (0x30, 0x60, 0xC0, 255))
    return str(out)


# --- project -----------------------------------------------------------------


class TestProject:
    def test_defaults(self, p: Project) -> None:
        assert p.fps == 30.0
        assert p.duration == 0.0
        assert p.size == (1920, 1080)
        assert p.canvas == "16:9"
        assert p.background == (10, 20, 30)

    def test_background_accepts_hex_and_named(self) -> None:
        assert Project("x", background="#ff8000").background == (255, 128, 0)
        assert Project("x", background="white").background == (255, 255, 255)

    def test_canvas_setter_changes_size(self, p: Project) -> None:
        p.canvas = "9:16"
        assert p.canvas == "9:16"
        assert p.size == (1080, 1920)

    def test_unknown_canvas_rejected(self, p: Project) -> None:
        with pytest.raises(ValueError):
            Project("x", canvas="17:4")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            p.canvas = "bogus"  # type: ignore[assignment]

    def test_background_setter(self, p: Project) -> None:
        p.background = (1, 2, 3)
        assert p.background == (1, 2, 3)
        p.background = "#804020"
        assert p.background == (128, 64, 32)

    def test_repr(self, p: Project) -> None:
        assert "1920" in repr(p)

    def test_save_and_load_round_trip(self, p: Project, tmp_path) -> None:
        track = p.add_track("sticker", name="BG")
        track.add(Solid("red"), start=0.0, duration=2.0)
        path = tmp_path / "doc.cutlass"
        p.save(str(path))

        loaded = Project.load(str(path))
        assert loaded.duration == pytest.approx(2.0)
        assert loaded.track("BG").kind == "sticker"
        assert len(loaded.track("BG").clips) == 1


# --- tracks ------------------------------------------------------------------


class TestTracks:
    def test_add_track_kinds(self, p: Project) -> None:
        for kind in ["video", "audio", "text", "sticker", "effect", "filter", "adjustment"]:
            assert p.add_track(kind).kind == kind  # type: ignore[arg-type]

    def test_unknown_kind_rejected(self, p: Project) -> None:
        with pytest.raises(ValueError):
            p.add_track("subtitle")  # type: ignore[arg-type]

    def test_tracks_are_ordered_and_insertable(self, p: Project) -> None:
        p.add_track("video", name="A")
        p.add_track("video", name="B")
        # The first video lane is the main track; nothing but audio may sit
        # below it, so an insert at index 0 clamps to just above it.
        p.add_track("video", name="above-main", index=0)
        assert [t.name for t in p.tracks] == ["A", "above-main", "B"]

    def test_first_video_track_is_main(self, p: Project) -> None:
        audio = p.add_track("audio")
        assert audio.main is False
        v1 = p.add_track("video", name="A")
        v2 = p.add_track("video", name="B")
        assert (v1.main, v2.main) == (True, False)
        v1.remove()
        assert v2.main is True

    def test_lookup_by_name(self, p: Project) -> None:
        p.add_track("video", name="Main")
        assert p.track("Main").name == "Main"
        with pytest.raises(ValueError):
            p.track("nope")
        p.add_track("audio", name="Main")
        with pytest.raises(ValueError):
            p.track("Main")  # ambiguous

    def test_flags_and_name(self, p: Project) -> None:
        t = p.add_track("video", name="old")
        t.name = "new"
        assert t.name == "new"
        t.enabled = False
        t.muted = True
        t.locked = True
        assert (t.enabled, t.muted, t.locked) == (False, True, True)

    def test_remove_makes_handle_stale(self, p: Project) -> None:
        t = p.add_track("video")
        t.remove()
        assert p.tracks == []
        with pytest.raises(CutlassError):
            _ = t.name

    def test_len_iter_end_clip_at(self, p: Project) -> None:
        t = p.add_track("sticker")
        t.add(Solid("red"), start=0.0, duration=1.0)
        t.add(Solid("blue"), start=2.0, duration=1.0)
        assert len(t) == 2
        assert [c.start for c in t] == [0.0, 2.0]
        assert t.end == pytest.approx(3.0)
        hit = t.clip_at(0.5)
        assert hit is not None and hit.start == 0.0
        assert t.clip_at(1.5) is None


# --- placement ---------------------------------------------------------------


class TestPlacement:
    def test_generated_content_requires_duration(self, p: Project) -> None:
        t = p.add_track("sticker")
        with pytest.raises(ValueError):
            t.add(Solid("red"), start=0.0)

    def test_non_positive_duration_rejected(self, p: Project) -> None:
        t = p.add_track("sticker")
        with pytest.raises(ValueError):
            t.add(Solid("red"), start=0.0, duration=0.0)
        with pytest.raises(ValueError):
            t.add(Solid("red"), start=0.0, duration=-1.0)

    def test_negative_start_rejected(self, p: Project) -> None:
        t = p.add_track("sticker")
        with pytest.raises(ValueError):
            t.add(Solid("red"), start=-1.0, duration=1.0)

    def test_overlap_raises(self, p: Project) -> None:
        t = p.add_track("sticker")
        t.add(Solid("red"), start=0.0, duration=2.0)
        with pytest.raises(OverlapError):
            t.add(Solid("blue"), start=1.0, duration=2.0)

    def test_track_kind_enforced(self, p: Project) -> None:
        video = p.add_track("video")
        text = p.add_track("text")
        sticker = p.add_track("sticker")
        with pytest.raises(TrackKindError):
            video.add(Solid("red"), start=0.0, duration=1.0)
        with pytest.raises(TrackKindError):
            sticker.add(Text("hi"), start=0.0, duration=1.0)
        with pytest.raises(TrackKindError):
            text.add(Solid("red"), start=0.0, duration=1.0)

    def test_append_butts_clips(self, p: Project) -> None:
        t = p.add_track("sticker")
        a = t.add(Solid("red"), start=0.0, duration=1.5)
        b = t.append(Solid("blue"), duration=1.0)
        assert b.start == pytest.approx(a.end)
        assert t.end == pytest.approx(2.5)

    def test_unknown_content_rejected(self, p: Project) -> None:
        t = p.add_track("sticker")
        with pytest.raises(ValueError):
            t.add("not content", start=0.0, duration=1.0)  # type: ignore[arg-type]

    def test_sticker_places_and_renders(self, p: Project) -> None:
        t = p.add_track("sticker")
        t.add(cutlass.Sticker("heart"), start=0.0, duration=2.0)
        frame = p.get_frame(0.5)
        h, w = frame.shape[0], frame.shape[1]
        # The bundled heart's red fill lands at the canvas center.
        assert frame[h // 2, w // 2][0] > 150

    def test_unknown_sticker_rejected(self) -> None:
        with pytest.raises(ValueError):
            cutlass.Sticker("not-a-sticker")

    def test_sticker_catalog_lists_bundled_pack(self) -> None:
        stickers = cutlass.stickers()
        ids = {s["id"] for s in stickers}
        assert {"heart", "star_spin"} <= ids
        animated = {s["id"] for s in stickers if s["animated"]}
        assert "star_spin" in animated

    def test_animation_catalog_lists_presets(self) -> None:
        animations = cutlass.animations()
        ids = {a["id"] for a in animations}
        assert {"fade_in", "fade_out", "pulse", "typewriter"} <= ids
        assert all(a["slot"] in {"in", "out", "combo"} for a in animations)

    def test_set_animation_on_clip(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=1.0)
        c.set_animation("in", "fade_in")
        assert c.animation_in == "fade_in"
        c.set_animation("combo", "pulse")
        assert c.animation_combo == "pulse"
        assert c.animation_in is None
        c.set_animation("combo")
        assert c.animation_combo is None


# --- clip structure ----------------------------------------------------------


class TestClipStructure:
    def test_timing(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=1.0, duration=2.0)
        assert c.start == pytest.approx(1.0)
        assert c.end == pytest.approx(3.0)
        assert c.duration == pytest.approx(2.0)
        assert c.media is None

    def test_track_backref(self, p: Project) -> None:
        t = p.add_track("sticker", name="S")
        c = t.add(Solid("red"), start=0.0, duration=1.0)
        assert c.track.name == "S"

    def test_split(self, p: Project) -> None:
        t = p.add_track("sticker")
        c = t.add(Solid("red"), start=0.0, duration=2.0)
        right = c.split(at=0.5)
        assert c.duration == pytest.approx(0.5)
        assert right.start == pytest.approx(0.5)
        assert right.end == pytest.approx(2.0)
        assert len(t) == 2

    def test_split_outside_clip_rejected(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        with pytest.raises(CutlassError):
            c.split(at=3.0)

    def test_trim(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=4.0)
        c.trim(start=1.0, end=3.0)
        assert (c.start, c.end) == (pytest.approx(1.0), pytest.approx(3.0))
        c.trim(end=2.0)  # keep start
        assert (c.start, c.end) == (pytest.approx(1.0), pytest.approx(2.0))
        c.trim(start=1.5)  # keep end
        assert (c.start, c.end) == (pytest.approx(1.5), pytest.approx(2.0))
        with pytest.raises(ValueError):
            c.trim(start=3.0, end=1.0)

    def test_move(self, p: Project) -> None:
        t = p.add_track("sticker")
        other = p.add_track("sticker")
        text = p.add_track("text")
        c = t.add(Solid("red"), start=0.0, duration=1.0)
        c.move(start=5.0)
        assert c.start == pytest.approx(5.0)
        c.move(start=0.0, track=other)
        assert len(t) == 0 and len(other) == 1
        with pytest.raises(TrackKindError):
            c.move(start=0.0, track=text)

    def test_delete_leaves_gap(self, p: Project) -> None:
        t = p.add_track("sticker")
        a = t.add(Solid("red"), start=0.0, duration=1.0)
        b = t.add(Solid("blue"), start=1.0, duration=1.0)
        a.delete()
        assert b.start == pytest.approx(1.0)
        with pytest.raises(CutlassError):
            _ = a.start

    def test_ripple_delete_closes_gap(self, p: Project) -> None:
        t = p.add_track("sticker")
        a = t.add(Solid("red"), start=0.0, duration=1.0)
        b = t.add(Solid("blue"), start=1.0, duration=1.0)
        a.ripple_delete()
        assert b.start == pytest.approx(0.0)


# --- transform ---------------------------------------------------------------


class TestTransform:
    def test_defaults(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=1.0)
        assert c.position == (0.0, 0.0)
        assert c.anchor == (0.5, 0.5)
        assert c.scale == 1.0
        assert c.rotation == 0.0
        assert c.opacity == 1.0

    def test_setters(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=1.0)
        c.position = (0.25, -0.1)
        c.anchor = (0.0, 0.0)
        c.scale = 0.5
        c.rotation = 15.0
        c.opacity = 0.85
        assert c.position == (pytest.approx(0.25), pytest.approx(-0.1))
        assert c.anchor == (0.0, 0.0)
        assert c.scale == pytest.approx(0.5)
        assert c.rotation == pytest.approx(15.0)
        assert c.opacity == pytest.approx(0.85)

    def test_invalid_values_rejected(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=1.0)
        with pytest.raises(CutlassError):
            c.opacity = 1.5
        with pytest.raises(CutlassError):
            c.scale = 0.0


# --- animation ---------------------------------------------------------------


class TestAnimation:
    def test_single_keyframe_requires_at(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        with pytest.raises(ValueError):
            c.animate(opacity=0.5)

    def test_no_properties_rejected(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        with pytest.raises(ValueError):
            c.animate()
        with pytest.raises(ValueError):
            c.animate(at=1.0)

    def test_batch_curve_and_sampling(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        c.animate(opacity=[(0.0, 0.0), (1.0, 1.0)])
        (pos, scale, rotation, opacity) = c.transform_at(0.5)
        assert pos == (0.0, 0.0)
        assert scale == 1.0
        assert rotation == 0.0
        assert opacity == pytest.approx(0.5, abs=0.05)

    def test_batch_pairs_may_be_unsorted(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        c.animate(opacity=[(1.0, 1.0), (0.0, 0.0)])
        assert c.transform_at(1.0)[3] == pytest.approx(1.0)

    def test_duplicate_times_rejected(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        with pytest.raises(ValueError):
            c.animate(opacity=[(0.5, 0.0), (0.5, 1.0)])

    def test_vector_and_multi_property(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        c.animate(scale=1.2, position=(0.1, 0.2), at=1.0)
        c.animate(position=[(0.0, (0.0, 0.0)), (2.0, (0.5, 0.5))])

    def test_easing_forms(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        c.animate(opacity=[(0.0, 0.0), (1.0, 1.0, "ease_in")], easing="ease_out")
        c.animate(scale=[(0.0, 1.0), (1.0, 2.0, (0.4, 0.0, 0.6, 1.0))])
        with pytest.raises(ValueError):
            c.animate(opacity=[(0.0, 0.0), (1.0, 1.0)], easing="bouncy")

    def test_keyframes_outside_clip_rejected(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        with pytest.raises(CutlassError):
            c.animate(opacity=0.5, at=5.0)

    def test_unknown_property_rejected(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        with pytest.raises(ValueError):
            c.animate(wobble=1.0, at=0.0)

    def test_non_numeric_at_rejected(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        with pytest.raises(ValueError):
            c.animate(opacity=0.5, at="soon")

    def test_remove_keyframe(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        c.animate(opacity=[(0.0, 0.0), (1.0, 1.0)])
        c.remove_keyframe("opacity", at=1.0)
        with pytest.raises(CutlassError):
            c.remove_keyframe("opacity", at=1.0)

    def test_clear_animation_flattens_to_clip_start_value(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        c.animate(opacity=[(0.0, 0.25), (1.0, 1.0)], scale=[(0.0, 1.0), (1.0, 2.0)])
        c.clear_animation("opacity", "scale")
        assert c.opacity == pytest.approx(0.25)
        assert c.scale == pytest.approx(1.0)
        assert c.transform_at(1.0)[3] == pytest.approx(0.25)
        with pytest.raises(ValueError):
            c.clear_animation()


# --- text --------------------------------------------------------------------


class TestText:
    def test_descriptor_validation(self) -> None:
        Text("ok", color=(255, 200, 100), align=("left", "top"))
        with pytest.raises(ValueError):
            Text("x", align=("diagonal", "top"))
        with pytest.raises(ValueError):
            Text("x", case="spongebob")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            Text("x", color="#12345")

    def test_descriptor_extras(self) -> None:
        Text(
            "styled",
            stroke=TextStroke("black", width=4.0),
            background=TextBackground("#00000080", radius=0.5),
            shadow=TextShadow("black", blur=0.2, distance=8.0),
        )

    def test_content_property(self, p: Project) -> None:
        c = p.add_track("text").add(Text("hello"), start=0.0, duration=1.0)
        assert c.text == "hello"
        c.text = "world"
        assert c.text == "world"

    def test_text_on_non_text_clip_rejected(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=1.0)
        with pytest.raises(ValueError):
            _ = c.text

    def test_set_style(self, p: Project) -> None:
        c = p.add_track("text").add(Text("hello"), start=0.0, duration=1.0)
        c.set_style(size=160, color="#ffcc00", italic=True, align=("left", "bottom"))
        c.set_style(stroke=TextStroke("white", 2.0))
        c.set_style(stroke=None)  # clears
        with pytest.raises(ValueError):
            c.set_style(kerning=1.0)
        assert c.text == "hello"  # content untouched

    def test_set_style_on_media_less_clip_kinds_only(self, p: Project) -> None:
        # set_style is for text/shape clips; a Solid has no style.
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=1.0)
        with pytest.raises(ValueError):
            c.set_style(color="blue")


# --- captions ----------------------------------------------------------------


SRT = """1
00:00:00,000 --> 00:00:01,500
first line

2
00:00:01,500 --> 00:00:03,000
second line
that wraps
"""


class TestCaptions:
    def add(self, p: Project, **kwargs) -> cutlass.Captions:
        return p.add_track("text").add_captions(
            [("hello there", 0.0, 1.5), ("general kenobi", 1.5, 2.0)], **kwargs
        )

    def test_cues_are_ordinary_text_clips(self, p: Project) -> None:
        captions = self.add(p)
        assert len(captions) == 2
        first, second = captions.cues
        assert first.text == "hello there"
        assert (first.start, first.duration) == (0.0, 1.5)
        assert second.start == 1.5
        # A cue trims, moves, and animates like any other clip.
        second.opacity = 0.5
        second.trim(end=3.0)
        assert second.duration == 1.5

    def test_dict_cues_accept_end_or_duration(self, p: Project) -> None:
        captions = p.add_track("text").add_captions(
            [
                {"text": "by duration", "start": 0.0, "duration": 1.0},
                {"text": "by end", "start": 1.0, "end": 2.5},
            ]
        )
        assert [c.duration for c in captions.cues] == [1.0, 1.5]

    def test_malformed_cues_rejected(self, p: Project) -> None:
        t = p.add_track("text")
        with pytest.raises(ValueError):
            t.add_captions([])
        with pytest.raises(ValueError):
            t.add_captions(["just a string"])
        with pytest.raises(ValueError):
            t.add_captions([("no time",)])  # type: ignore[list-item]
        with pytest.raises(ValueError):
            t.add_captions([("zero length", 0.0, 0.0)])
        with pytest.raises(ValueError):
            t.add_captions([{"start": 0.0, "duration": 1.0}])

    def test_captions_need_a_text_lane(self, p: Project) -> None:
        with pytest.raises(CutlassError):
            p.add_track("video").add_captions([("nope", 0.0, 1.0)])

    def test_project_lists_groups_and_clip_points_back(self, p: Project) -> None:
        captions = self.add(p, label="Intro")
        assert [c.label for c in p.captions] == ["Intro"]
        cue = captions.cues[0]
        assert cue.caption is not None
        assert cue.caption.label == "Intro"
        plain = p.add_track("text").add(Text("title"), start=10.0, duration=1.0)
        assert plain.caption is None

    def test_template_styles_every_line(self, p: Project) -> None:
        ids = [t["id"] for t in cutlass.caption_templates()]
        assert "karaoke_pop" in ids
        captions = self.add(p, template="karaoke_pop")
        assert captions.template == "karaoke_pop"
        captions.template = "clean"
        assert captions.template == "clean"
        with pytest.raises(ValueError):
            self.add(p, template="neon_wobble")

    def test_style_and_layout_patch(self, p: Project) -> None:
        captions = self.add(p)
        captions.style(size=96, fill="#ffcc00", uppercase=True, position_y=0.2)
        captions.layout(max_chars_per_line=20, max_lines=1, min_duration=0.5)
        captions.label = "Captions v2"
        assert captions.label == "Captions v2"
        with pytest.raises(CutlassError):
            captions.style(scale=50.0)
        with pytest.raises(CutlassError):
            captions.layout(max_chars_per_line=2)

    def test_highlight_modes(self, p: Project) -> None:
        captions = self.add(p)
        captions.highlight("word", fill="#ffd800", plate="#000000c8", scale=1.15)
        captions.highlight("line")
        captions.highlight("off")
        with pytest.raises(ValueError):
            captions.highlight("strobe")  # type: ignore[arg-type]

    def test_import_subtitles_round_trips_through_srt(self, p: Project, tmp_path) -> None:
        path = tmp_path / "dialogue.srt"
        path.write_text(SRT)
        captions = p.add_track("text").import_subtitles(str(path))
        assert len(captions) == 2
        assert captions.label == "dialogue.srt"
        assert captions.cues[0].text == "first line"
        # The file's own line breaks survive the trip.
        assert captions.cues[1].text == "second line\nthat wraps"

        out = tmp_path / "out.srt"
        text = captions.to_srt(str(out))
        assert out.read_text() == text
        # Cues are separated by the layout's minimum gap, so the first line ends
        # a frame early rather than touching the second.
        assert text.startswith("1\n00:00:00,000 --> 00:00:01,4")
        assert "second line\nthat wraps" in text
        assert captions.to_vtt().startswith("WEBVTT")

        # The sidecar reads back as the same lines.
        reimported = p.add_track("text").import_subtitles(str(out))
        assert [c.text for c in reimported.cues] == [c.text for c in captions.cues]

    def test_import_offsets_and_rewraps(self, p: Project, tmp_path) -> None:
        path = tmp_path / "dialogue.srt"
        path.write_text(SRT)
        captions = p.add_track("text").import_subtitles(
            str(path), start=10.0, rewrap=True, label="Reel"
        )
        assert captions.cues[0].start == 10.0
        assert "\n" not in captions.cues[1].text
        assert captions.label == "Reel"

    def test_import_rejects_missing_and_empty_files(self, p: Project, tmp_path) -> None:
        t = p.add_track("text")
        with pytest.raises(CutlassError):
            t.import_subtitles(str(tmp_path / "nope.srt"))
        empty = tmp_path / "empty.srt"
        empty.write_text("")
        with pytest.raises(CutlassError):
            t.import_subtitles(str(empty))

    def test_remove_takes_the_lines_with_it(self, p: Project) -> None:
        track = p.add_track("text")
        captions = track.add_captions([("bye", 0.0, 1.0)])
        captions.remove()
        assert len(track) == 0
        assert p.captions == []
        with pytest.raises(CutlassError):
            captions.label  # noqa: B018 — a stale handle raises


# --- shapes ------------------------------------------------------------------


class TestShapes:
    def test_all_shapes_place(self, p: Project) -> None:
        t = p.add_track("sticker")
        shapes = [
            Rect(width=400, height=300, color="#ff0055", corner_radius=24),
            Ellipse(width=200, height=100),
            Polygon(6, color=(0, 200, 100)),
            Star(5, inner_ratio=0.4),
            Line(length=300, thickness=6),
            Arrow(),
            Heart(color="red"),
        ]
        for i, shape in enumerate(shapes):
            t.add(shape, start=float(i), duration=1.0)
        assert len(t) == len(shapes)

    def test_stroke(self, p: Project) -> None:
        t = p.add_track("sticker")
        c = t.add(
            Rect(width=200, height=200, stroke=ShapeStroke("black", width=12.0)),
            start=0.0,
            duration=1.0,
        )
        c.set_style(stroke_width=4.0, stroke_color="#00ffaa")
        c.animate(stroke_width=[(0.0, 4.0), (1.0, 16.0)])

    def test_stroke_params_require_stroke(self, p: Project) -> None:
        c = p.add_track("sticker").add(Rect(), start=0.0, duration=1.0)
        with pytest.raises(CutlassError):
            c.set_style(stroke_width=4.0)

    def test_invalid_geometry_rejected_at_add(self, p: Project) -> None:
        t = p.add_track("sticker")
        with pytest.raises(CutlassError):
            t.add(Polygon(2), start=0.0, duration=1.0)
        with pytest.raises(CutlassError):
            t.add(Star(5, inner_ratio=1.5), start=0.0, duration=1.0)

    def test_set_style_constants(self, p: Project) -> None:
        c = p.add_track("sticker").add(Rect(width=400, height=300), start=0.0, duration=5.0)
        c.set_style(color="#00ffaa", width=800.0, height=100.0, corner_radius=12.0)
        with pytest.raises(ValueError):
            c.set_style(sides=8)

    def test_shape_animation(self, p: Project) -> None:
        c = p.add_track("sticker").add(Rect(width=400, height=300), start=0.0, duration=5.0)
        c.animate(width=800.0, at=1.0)
        c.animate(color=[(0.0, "#ff0000"), (2.0, (0, 0, 255))])
        c.clear_animation("width", "color")

    def test_inner_ratio_only_on_stars(self, p: Project) -> None:
        star = p.add_track("sticker").add(Star(5), start=0.0, duration=1.0)
        star.animate(inner_ratio=[(0.0, 0.2), (1.0, 0.8)])
        rect = p.add_track("sticker").add(Rect(), start=0.0, duration=1.0)
        with pytest.raises(CutlassError):
            rect.animate(inner_ratio=0.5, at=0.0)

    def test_shape_params_rejected_on_text(self, p: Project) -> None:
        c = p.add_track("text").add(Text("hi"), start=0.0, duration=1.0)
        with pytest.raises(CutlassError):
            c.animate(width=100.0, at=0.0)


# --- effects & catalogs --------------------------------------------------------


class TestEffects:
    def test_catalogs(self) -> None:
        effects = {e["id"]: e for e in cutlass.effects()}
        assert "gaussian_blur" in effects
        radius = next(p for p in effects["gaussian_blur"]["params"] if p["name"] == "radius")
        assert radius["max"] > radius["min"]
        assert "crossfade" in {t["id"] for t in cutlass.transitions()}

    def test_add_get_set(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        fx = c.add_effect("gaussian_blur", radius=8.0)
        assert fx["radius"] == pytest.approx(8.0)
        fx["radius"] = 16.0
        assert fx["radius"] == pytest.approx(16.0)
        assert len(c.effects) == 1

    def test_unknown_ids_rejected(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        with pytest.raises(CutlassError):
            c.add_effect("motion_blur_9000")
        fx = c.add_effect("gaussian_blur")
        with pytest.raises(ValueError):
            fx["oomph"] = 1.0

    def test_out_of_range_param_rejected(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        fx = c.add_effect("gaussian_blur")
        with pytest.raises(CutlassError):
            fx["radius"] = 1e9

    def test_animate(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        fx = c.add_effect("gaussian_blur")
        fx.animate(radius=[(0.0, 16.0), (2.0, 0.0)])
        fx.animate(radius=8.0, at=1.0)
        with pytest.raises(ValueError):
            fx.animate(radius=8.0)  # missing at=
        with pytest.raises(ValueError):
            fx.animate()

    def test_remove(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=2.0)
        fx = c.add_effect("gaussian_blur")
        c.remove_effect(fx)
        assert c.effects == []
        with pytest.raises(CutlassError):
            fx["radius"] = 1.0  # stale


# --- media -------------------------------------------------------------------


class TestMedia:
    def test_import_probes(self, media_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        assert m.kind == "video"
        assert m.duration == pytest.approx(2.0, abs=0.2)
        assert m.size == (1920, 1080)
        assert m.fps == pytest.approx(30.0, abs=0.5)
        assert "video" in repr(m)
        assert len(p.media) == 1

    def test_import_is_deduplicated(self, media_path: str) -> None:
        p = Project("m", fps=30)
        p.import_media(media_path)
        p.import_media(media_path)
        assert len(p.media) == 1

    def test_import_rejects_missing_and_corrupt(self, tmp_path) -> None:
        p = Project("m", fps=30)
        with pytest.raises(MediaError):
            p.import_media(str(tmp_path / "missing.mp4"))
        corrupt = tmp_path / "frame.png"
        corrupt.write_bytes(b"\x89PNG\r\n")
        with pytest.raises(MediaError):
            p.import_media(str(corrupt))

    def test_import_still_image(self, image_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(image_path)
        assert m.kind == "image"
        assert m.duration == pytest.approx(5.0, abs=0.05)
        assert m.size == (64, 48)
        assert m.has_audio is False
        assert "image" in repr(m)
        assert len(p.media) == 1

    def test_still_placement_allows_long_duration(self, image_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(image_path)
        clip = p.add_track("video").add(m, start=0.0, duration=8.0)
        assert clip.duration == pytest.approx(8.0)

    def test_still_renders(self, image_path: str) -> None:
        p = Project("m", fps=30, canvas="16:9", background=(0, 0, 0))
        m = p.import_media(image_path)
        p.add_track("video").add(m, start=0.0, duration=2.0)
        frame = p.get_frame(1.0)
        h, w = frame.shape[:2]
        r, g, b, a = frame[h // 2, w // 2]
        assert (r, g, b, a) == (0x30, 0x60, 0xC0, 255)

    def test_place_full_and_shorthand_duration(self, media_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        v = p.add_track("video")
        full = v.add(m, start=0.0)
        assert full.duration == pytest.approx(m.duration, abs=0.05)
        short = v.add(m, start=5.0, duration=1.0)
        assert short.duration == pytest.approx(1.0)
        assert short.source_start == pytest.approx(0.0)
        assert short.source_duration == pytest.approx(1.0)

    def test_subclip_and_slicing(self, media_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        v = p.add_track("video")
        a = v.add(m.subclip(0.5, 1.5), start=0.0)
        assert a.duration == pytest.approx(1.0)
        assert a.source_start == pytest.approx(0.5)
        b = v.add(m[0.5:1.5], start=2.0)
        assert b.duration == pytest.approx(1.0)
        c = v.add(m[:1], start=4.0)
        assert c.duration == pytest.approx(1.0)
        d = v.add(m[-0.5:], start=6.0)
        assert d.duration == pytest.approx(0.5)
        assert d.source_start == pytest.approx(m.duration - 0.5, abs=0.05)

    def test_slice_step_rejected(self, media_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        with pytest.raises(ValueError):
            _ = m[0:1:2]

    def test_out_of_range_window_rejected_at_add(self, media_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        v = p.add_track("video")
        with pytest.raises(MediaError):
            v.add(m.subclip(0.0, 99.0), start=0.0)
        with pytest.raises(ValueError):
            v.add(m.subclip(1.5, 0.5), start=0.0)

    def test_media_clip_kind_checked(self, media_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        with pytest.raises(TrackKindError):
            p.add_track("text").add(m, start=0.0)

    def test_media_across_projects_rejected(self, media_path: str) -> None:
        a = Project("a", fps=30)
        b = Project("b", fps=30)
        m = a.import_media(media_path)
        with pytest.raises(CutlassError):
            b.add_track("video").add(m, start=0.0)

    def test_split_partitions_source(self, media_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        c = p.add_track("video").add(m, start=0.0)
        right = c.split(at=1.0)
        assert c.source_start == pytest.approx(0.0)
        assert c.source_duration == pytest.approx(1.0, abs=0.05)
        assert right.source_start == pytest.approx(1.0, abs=0.05)

    def test_speed(self, media_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        c = p.add_track("video").add(m, start=0.0)
        original = c.duration
        c.set_speed(2.0)
        assert c.speed == pytest.approx(2.0)
        assert c.duration == pytest.approx(original / 2, abs=0.05)
        c.set_speed(0.5, reverse=True)
        assert c.reversed is True
        with pytest.raises(ValueError):
            c.set_speed(0.0)

    def test_speed_requires_media(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=1.0)
        with pytest.raises(CutlassError):
            c.set_speed(2.0)

    def test_volume_and_fades(self, media_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        c = p.add_track("video").add(m, start=0.0)
        c.volume = 0.8
        assert c.volume == pytest.approx(0.8)
        c.fade_in = 0.5
        c.fade_out = 0.25
        assert c.fade_in == pytest.approx(0.5, abs=1 / 30)  # frame-quantized
        assert c.fade_out == pytest.approx(0.25, abs=1 / 30)
        assert c.volume == pytest.approx(0.8)  # fades didn't clobber gain
        with pytest.raises(CutlassError):
            c.volume = 100.0
        with pytest.raises(CutlassError):
            c.fade_in = 99.0  # longer than the clip

    def test_volume_envelope(self, media_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        c = p.add_track("video").add(m, start=0.0)
        c.animate(volume=[(0.0, 1.0), (1.0, 0.2), (2.0, 1.0)])
        c.clear_animation("volume")
        assert c.volume == pytest.approx(1.0)

    def test_volume_requires_media(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("red"), start=0.0, duration=1.0)
        with pytest.raises(CutlassError):
            c.volume = 0.5

    def test_crop(self, media_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        c = p.add_track("video").add(m, start=0.0)
        c.crop(x=0.1, y=0.0, w=0.8, h=1.0, flip_h=True)
        with pytest.raises(CutlassError):
            c.crop(w=0.0)

    def test_transitions(self, media_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        v = p.add_track("video")
        a = v.add(m.subclip(0.0, 1.0), start=0.0)
        v.append(m.subclip(1.0, 2.0))
        a.transition("crossfade", duration=0.4)
        a.remove_transition()
        with pytest.raises(CutlassError):
            a.transition("teleport")
        lone = v.add(m.subclip(0.0, 0.5), start=5.0)
        with pytest.raises(CutlassError):
            lone.transition("crossfade")

    def test_remove_media_guards_references(self, media_path: str) -> None:
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        c = p.add_track("video").add(m, start=0.0)
        with pytest.raises(MediaError):
            p.remove_media(m)
        c.delete()
        p.remove_media(m)
        assert p.media == []
        with pytest.raises(CutlassError):
            _ = m.duration  # stale


# --- rendering ---------------------------------------------------------------


class TestRendering:
    def test_background_frame(self, p: Project) -> None:
        frame = p.get_frame(0.0)
        h, w = frame.shape[:2]
        assert (w, h) == p.size
        assert frame.dtype.name == "uint8"
        assert tuple(frame[h // 2, w // 2]) == (10, 20, 30, 255)

    def test_solid_composites_over_background(self, p: Project) -> None:
        p.add_track("sticker").add(Solid("#3060c0"), start=0.0, duration=1.0)
        frame = p.get_frame(0.5)
        h, w = frame.shape[:2]
        r, g, b, a = frame[h // 2, w // 2]
        assert (r, g, b, a) == (0x30, 0x60, 0xC0, 255)

    def test_opacity_animation_renders(self, p: Project) -> None:
        c = p.add_track("sticker").add(Solid("white"), start=0.0, duration=2.0)
        c.animate(opacity=[(0.0, 0.0), (2.0, 1.0)])
        h, w = p.size[1], p.size[0]
        start = p.get_frame(0.0)[h // 2, w // 2]
        end = p.get_frame(1.9)[h // 2, w // 2]
        assert int(end[0]) > int(start[0])  # brightens as opacity ramps

    def test_export_and_reimport(self, media_path: str) -> None:
        # media_path itself was produced by export(); assert it round-trips.
        p = Project("m", fps=30)
        m = p.import_media(media_path)
        c = p.add_track("video").add(m, start=0.0)
        frame = p.get_frame(1.0)
        h, w = frame.shape[:2]
        r, g, b, _ = frame[h // 2, w // 2]
        # The sample is a flat #3060c0 solid; allow encoder drift.
        assert abs(int(r) - 0x30) < 16
        assert abs(int(g) - 0x60) < 16
        assert abs(int(b) - 0xC0) < 16
        assert c.duration == pytest.approx(2.0, abs=0.2)


# --- errors ------------------------------------------------------------------


class TestErrors:
    def test_hierarchy(self) -> None:
        for exc in (OverlapError, TrackKindError, MediaError, cutlass.RenderError):
            assert issubclass(exc, CutlassError)
        assert issubclass(CutlassError, Exception)
