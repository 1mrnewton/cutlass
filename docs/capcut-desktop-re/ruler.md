# CapCut timeline ruler

CapCut 9.6.0 (`com.lemon.lvoverseas`). The timeline ruler is `VECTimelineRuler`, a `QQuickPaintedItem` in `libVECreator.dylib` (arm64, sha256 `f1d55362efed98183ce6fed889f9823a5b2934c7c6ea17268f498694e922204f`). The step table is `TimelineRulerAccuracyConfigPrivate`. The binary still has the path `TimeLine/ruler/timeline_ruler_item.cpp`. This is static disassembly of those functions. `PlayerRulerItem` is the preview ruler and is not this one.

## Time and zoom

Draft time is microseconds. `lvve::FrameRate::FrameInterval` is microseconds per frame. The ruler's `pixelSize` is pixels per microsecond, so a span of `us` microseconds is `us * pixelSize` pixels wide.

`setPixelSize` ignores a change that is tiny next to the current value (the same 1e12 relative test Qt uses), then asks the accuracy table for a new step and repaints. `setOriginOffset` is the scroll, in pixels. A negative offset means time zero is off the left of the ruler.

Frame rates, from `InitFramerate*`:

| Id | Rate | Whole-frame count used for steps |
| --- | --- | --- |
| 1 | 23.976 | 24 |
| 2 | 24 | 24 |
| 3 | 25 | 25 |
| 4 | 29.97 | 30 |
| 5 | 30 | 30 |
| 6 | 50 | 50 |
| 7 | 59.94 | 60 |
| 8 | 60 | 60 |

## Which step is drawn

`AccuracyType(frameRateId, pixelSize)` walks that rate's steps from coarse to fine and keeps the first whose pixel-size threshold is greater than or equal to the current zoom. Zooming out selects a coarser step. Zooming in walks further down the list. Past the last threshold, the fallback is 2 frames, except 25 fps, which stops at 5 frames.

The thresholds are the zoom at which that labeled step is about 300 pixels wide. The 30-hour family uses 2.6e-9 instead of an exact 300, so those steps are 280.8 pixels at the switch.

Clock steps, same for every frame rate. Thresholds are pixels per microsecond.

| Step | Threshold | Seconds in the label |
| --- | --- | --- |
| 30 h | 2.6e-9 | 108000 |
| 10 h | 8e-9 | 36000 |
| 5 h | 1.6e-8 | 18000 |
| 3 h | 2.6e-8 | 10800 |
| 2 h | 4e-8 | 7200 |
| 1 h | 8e-8 | 3600 |
| 30 min | 1.6e-7 | 1800 |
| 10 min | 5e-7 | 600 |
| 5 min | 1e-6 | 300 |
| 3 min | 1.6e-6 | 180 |
| 2 min | 2.5e-6 | 120 |
| 1 min | 5e-6 | 60 |
| 30 s | 1e-5 | 30 |
| 10 s | 3e-5 | 10 |
| 5 s | 6e-5 | 5 |
| 3 s | 1e-4 | 3 |
| 2 s | 1.5e-4 | 2 |
| 1 s | 3e-4 | 1 |

Frame steps are a whole number of the rate's nominal frames. Only some of them are installed for each rate. Thresholds below are pixels per microsecond.

| Frames | 23.976 | 24 | 25 | 29.97 | 30 | 50 | 59.94 | 60 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | | | | | | | 0.0006 | 0.0006 |
| 25 | | | | | | 0.0006 | | |
| 20 | | | | | | | 0.0008 | 0.0008 |
| 15 | | | | 0.0006 | 0.0006 | | 0.00085 | 0.00085 |
| 12 | 0.0006 | 0.0006 | | | | | 0.0009 | |
| 10 | | | | 0.0008 | 0.0008 | 0.0009 | 0.001 | 0.001 |
| 8 | 0.0008 | 0.0008 | | | | | | |
| 6 | 0.0009 | | | 0.0009 | | | 0.0013 | |
| 5 | | | 0.005 | 0.001 | 0.001 | 0.005 | 0.003 | 0.003 |
| 4 | 0.001 | 0.001 | | | | | 0.006 | |
| 3 | 0.002 | 0.002 | | 0.0013 | 0.0013 | | 0.009 | 0.009 |
| 2 | 0.0048 | 0.0048 | | 0.006 | 0.006 | 0.01 | 0.012 | 0.012 |

A blank cell is not a step for that rate. 25 fps has no 2-frame step.

## Minor ticks

A labeled step is split into minor ticks. Clock steps use one count for the whole rate: 12 at 23.976 and 24, 5 at 25, and 10 at 29.97, 30, 50, 59.94, and 60. One second at 24 fps is therefore a minor tick every 2 frames. One second at 30 fps is a minor tick every 3 frames.

Frame steps override that count:

| Frames in the label | Minor ticks inside it |
| --- | --- |
| 30 | 10 |
| 25 | 5 |
| 20 | 10 |
| 15 | 5 |
| 12 | 6 |
| 10 | 10 |
| 8 | 4 |
| 6 | 6 |
| 5 | 5 |
| 4 | 4 |
| 3 | 3 |
| 2 | 2 |

The finest steps are one minor tick per frame. An 8-frame label is a minor tick every 2 frames.

## How it is painted

`paint` returns immediately when the minor-tick spacing is zero. Otherwise it turns on antialiasing and draws two passes, each capped at 1001 marks.

The first pass draws vertical lines. Tick zero sits at `originOffset`. Each next tick moves right by one minor spacing. A tick whose index divides the minor count is a major tick: the long-line color, and the full ruler height. The others use the short-line color and half the height. Lines start at the top of the item. Ticks left of x = 0 are skipped. The loop stops at the item's width. Nothing in the painter checks the end of the cut, so a scrolled ruler keeps drawing marks past the last clip.

The second pass draws labels on the major ticks only. The text starts 4 pixels to the right of the tick, and its baseline is 1 pixel above the bottom of the ruler.

## Labels

The usual label is clock time, from `index * majorMicroseconds / 1000000`. At an hour or more the format is `%1:%2:%3`, zero-padded `HH:MM:SS`. Under an hour it is `%1:%2`, zero-padded `MM:SS`.

On a frame step, a major tick that lands on a whole second still uses that clock text. A major tick inside the second uses `%1f`, and the number is the frame offset of that tick. A 12-frame step at 24 fps therefore alternates `MM:SS` and `12f`.

Full timecode is a separate path, `RulerTextFullTimecodeByFrameIndex`. `UpdateUiInfo` turns it on only when the global timecode style is not the literal `HH:MM:SS:FF` and the rate's denominator is not 1. The string is `%1:%2:%3:%4`, zero-padded `HH:MM:SS:FF`. `setFramerate` also takes a drop-frame flag. When that flag is set and the denominator is at least 2, the format becomes `%1:%2:%3;%4` (`HH:MM:SS;FF`) and the frame count uses SMPTE drop-frame: 29.97 drops 2 frames a minute, 59.94 drops 4, except on every tenth minute. The other six rates drop nothing. Who writes the global style string was not traced.

The prototype ruler uses this clock path at the project frame rate, which starts at 30 fps. It draws the sequence clock from `docs/project.md`, and pixels per second is the pixel size times 1,000,000. Drop frame is a project setting. On 29.97 and 59.94 it draws `HH:MM:SS;FF` and skips the SMPTE frame numbers. `HH:MM:SS:FF` labels are not drawn.
