import { clipAt, formatTimecode, scenes, type Clip, type FrameRate } from "./model"
import { Icon } from "./icons"
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
  const picture = clipAt(clips, time, "picture")
  const title = clipAt(clips, time, "title")
  const generated = clipAt(clips, time, "generated")
  const wash = scenes[picture?.scene ?? picture?.id ?? "ceremony"] ?? scenes.ceremony
  const lower = Boolean(title && title.name.includes("·"))

  return (
    <div className="preview">
      <div
        className="preview-frame"
        style={{ background: `linear-gradient(155deg, ${wash[0]}, ${wash[1]} 46%, ${wash[2]})` }}
      >
        <div className="shot-label">
          {picture?.name ?? "Empty"} · {formatTimecode(time, rate, dropFrame)}
        </div>
        {generated && <div className="pip">Generated shot</div>}
        {title && !lower && (
          <div className="title-card">
            <p>
              {title.name}
              <small>{title.id === "end" ? "End card" : "Title"}</small>
            </p>
          </div>
        )}
        {title && lower && <div className="lower-third">{title.name}</div>}
      </div>
    </div>
  )
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
