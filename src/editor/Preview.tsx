import { clipAt, formatTimecode, scenes, type Clip, type FrameRate } from "./model"
import { Icon } from "./icons"
import { tintHex, useLook, type Look } from "./look"
import { Tip } from "./Tooltip"

export function Preview({
  clips,
  time,
  rate,
  dropFrame,
}: {
  clips: Clip[]
  time: number
  rate: FrameRate
  dropFrame: boolean
}) {
  const { looks } = useLook()
  const picture = clipAt(clips, time, "picture")
  const title = clipAt(clips, time, "title")
  const generated = clipAt(clips, time, "generated")
  const filter = clipAt(clips, time, "filter")
  const effect = clipAt(clips, time, "effect")
  const adjust = clipAt(clips, time, "adjust")
  const sound = clipAt(clips, time, "sound")
  const sticker = clipAt(clips, time, "sticker")
  const wash = scenes[picture?.scene ?? picture?.id ?? "ceremony"] ?? scenes.ceremony
  const grade = mergeLook(
    looks[picture?.id ?? ""],
    implied(adjust),
    looks[adjust?.id ?? ""],
    implied(filter),
    looks[filter?.id ?? ""],
    implied(effect),
    looks[effect?.id ?? ""],
  )
  const tinted = wash.map((stop) => tintHex(stop, grade)) as [string, string, string]
  const lower = Boolean(title && title.name.includes("·"))
  const textLook = title ? looks[title.id] : undefined
  const level = sound ? Math.max(0, Math.min(100, ((looks[sound.id]?.volume ?? 0) + 60) / 72 * 100)) : 0

  return (
    <div className="preview">
      <div
        className="preview-frame"
        style={{ background: `linear-gradient(155deg, ${tinted[0]}, ${tinted[1]} 46%, ${tinted[2]})` }}
      >
        <div className="shot-label">
          {picture?.name ?? "Empty"} · {formatTimecode(time, rate, dropFrame)}
        </div>
        {generated && <div className="pip">Generated shot</div>}
        {title && !lower && (
          <div className="title-card" data-bubble={textLook?.bubble ?? "none"} data-effect={textLook?.textEffect ?? "none"}>
            <p>
              {title.name}
              <small>{title.assetId?.startsWith("caption:") ? "Caption" : title.id === "end" ? "End card" : "Title"}</small>
            </p>
          </div>
        )}
        {title && lower && <div className="lower-third">{title.name}</div>}
        {sticker && (
          <div className="sticker-mark" style={{ opacity: (looks[sticker.id]?.opacity ?? 100) / 100 }}>
            {sticker.name}
          </div>
        )}
        {sound && (
          <div className="sound-meter" aria-label={`${sound.name} volume`}>
            <span style={{ width: `${level}%` }} />
          </div>
        )}
      </div>
    </div>
  )
}

function implied(clip: Clip | null): Look {
  if (!clip) return {}
  if (clip.tone === "filter") return { amount: 60, temp: clip.name === "Cool" ? -30 : 20 }
  if (clip.tone === "effect") {
    return {
      amount: 70,
      temp: clip.name === "Warm film" ? 24 : 0,
      sat: clip.name === "Fade grain" ? -28 : 0,
      bright: clip.name === "Bloom" ? 16 : 0,
    }
  }
  if (clip.tone === "adjust") {
    return {
      temp: clip.name === "Temperature" ? 35 : 0,
      bright: clip.name === "Exposure" ? 20 : 0,
      sat: clip.name === "Contrast" ? 25 : 0,
    }
  }
  return {}
}

function mergeLook(...parts: (Look | undefined)[]) {
  return parts.reduce<Look>((all, part) => ({ ...all, ...part }), {})
}

export function Transport({
  time,
  duration,
  rate,
  dropFrame,
  playing,
  empty,
  onTogglePlay,
  onStill,
}: {
  time: number
  duration: number
  rate: FrameRate
  dropFrame: boolean
  playing: boolean
  empty: boolean
  onTogglePlay: () => void
  onStill: () => void
}) {
  const timecode = (seconds: number) => formatTimecode(seconds, rate, dropFrame)
  return (
    <div className="transport">
      <span className="transport-time">
        {timecode(time)} <span>/ {timecode(duration)}</span>
      </span>
      <Tip label={playing ? "Pause" : "Play"}>
        <button type="button" className="play" aria-label={playing ? "Pause" : "Play"} disabled={empty} onClick={onTogglePlay}>
          <Icon name={playing ? "pause" : "play"} />
        </button>
      </Tip>
      <Tip label="Export still frame">
        <button type="button" className="icon-btn" aria-label="Export still frame" disabled={empty} onClick={onStill}>
          <Icon name="camera" />
        </button>
      </Tip>
    </div>
  )
}
