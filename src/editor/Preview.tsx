import { clipAt, formatTime, scenes, type Clip } from "./model"
import { Icon } from "./icons"

export function Preview({
  clips,
  time,
  note,
}: {
  clips: Clip[]
  time: number
  note: string | null
}) {
  const picture = clipAt(clips, time, "picture")
  const title = clipAt(clips, time, "title")
  const generated = clipAt(clips, time, "generated")
  const wash = scenes[picture?.id ?? "ceremony"] ?? scenes.ceremony
  const lower = Boolean(title && title.name.includes("·"))

  return (
    <div className="preview">
      <div
        className="preview-frame"
        style={{ background: `linear-gradient(155deg, ${wash[0]}, ${wash[1]} 46%, ${wash[2]})` }}
      >
        <div className="shot-label">
          {picture?.name ?? "Empty"} · {formatTime(time)}
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
        {note && <div className="toast">{note}</div>}
      </div>
    </div>
  )
}

export function Transport({
  time,
  duration,
  playing,
  onTogglePlay,
  onSeek,
}: {
  time: number
  duration: number
  playing: boolean
  onTogglePlay: () => void
  onSeek: (time: number) => void
}) {
  return (
    <div className="transport">
      <button
        type="button"
        className="icon-btn"
        aria-label="Back 5 seconds"
        onClick={() => onSeek(Math.max(0, time - 5))}
      >
        <Icon name="back" />
      </button>
      <button type="button" className="play" aria-label={playing ? "Pause" : "Play"} onClick={onTogglePlay}>
        <Icon name={playing ? "pause" : "play"} />
      </button>
      <button
        type="button"
        className="icon-btn"
        aria-label="Forward 5 seconds"
        onClick={() => onSeek(Math.min(duration, time + 5))}
      >
        <Icon name="forward" />
      </button>
      <span className="transport-time">
        {formatTime(time)} <span>/ {formatTime(duration)}</span>
      </span>
    </div>
  )
}
