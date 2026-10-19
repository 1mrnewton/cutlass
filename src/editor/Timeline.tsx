import {
  formatMark,
  formatTime,
  rulerMarks,
  type Clip,
  type Track,
} from "./model"

export function Timeline({
  tracks,
  clips,
  duration,
  time,
  selectedId,
  silencesRemoved,
  onSeek,
  onSelect,
}: {
  tracks: Track[]
  clips: Clip[]
  duration: number
  time: number
  selectedId: string | null
  silencesRemoved: boolean
  onSeek: (time: number) => void
  onSelect: (id: string) => void
}) {
  return (
    <section className="timeline" aria-label="Timeline">
      <div className="tl-bar">
        <span>Timeline</span>
        <span className="tl-fit">Fit</span>
        <span className="tl-time">{formatTime(time)}</span>
      </div>
      <div className="tl-body">
        <div className="tl-labels">
          <div className="ruler-space" />
          {tracks.map((track) => (
            <div key={track.id} className="track-label">
              {track.label}
            </div>
          ))}
        </div>
        <div
          className="sheet"
          onClick={(event) => {
            const rect = event.currentTarget.getBoundingClientRect()
            const ratio = (event.clientX - rect.left) / rect.width
            onSeek(Math.min(duration, Math.max(0, ratio * duration)))
          }}
        >
          <div className="ruler">
            {rulerMarks.map((mark) => (
              <span key={mark} className="tick" style={{ left: `${(mark / duration) * 100}%` }}>
                {mark % 10 === 0 ? formatMark(mark) : ""}
              </span>
            ))}
          </div>
          {tracks.map((track) => (
            <div key={track.id} className="lane">
              {clips
                .filter((clip) => clip.trackId === track.id)
                .map((clip) => (
                  <button
                    key={clip.id}
                    type="button"
                    className="clip"
                    data-tone={clip.tone}
                    data-selected={selectedId === clip.id ? "true" : "false"}
                    style={{
                      left: `calc(${(clip.start / duration) * 100}% + 1px)`,
                      width: `calc(${(clip.duration / duration) * 100}% - 2px)`,
                    }}
                    onClick={(event) => {
                      event.stopPropagation()
                      onSelect(clip.id)
                    }}
                  >
                    <span>{clip.name}</span>
                    {silencesRemoved && clip.tone === "picture" && (
                      <>
                        <i className="notch" style={{ left: "22%" }} />
                        <i className="notch" style={{ left: "48%" }} />
                        <i className="notch" style={{ left: "71%" }} />
                      </>
                    )}
                  </button>
                ))}
            </div>
          ))}
          <div className="playhead" style={{ left: `${(time / duration) * 100}%` }} />
        </div>
      </div>
    </section>
  )
}
