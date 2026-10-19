import { AssetBrowser } from "./AssetBrowser"
import { CommandBox } from "./CommandBox"
import { Blade, Icon } from "./icons"
import { findClip, formatTime, tools, type EditorApi } from "./model"
import { Preview, Transport } from "./Preview"
import { Timeline } from "./Timeline"

export function DesktopEditor({ api }: { api: EditorApi }) {
  const selected = findClip(api.project.clips, api.selectedId)

  return (
    <div className="desktop">
      <header className="topbar">
        <div className="brand">
          <Blade />
          <span>Cutlass</span>
        </div>
        <div className="project-name">Amina & Jules</div>
        <div className="spacer" />
        <div className="timecode">
          {formatTime(api.time)} <span>/ {formatTime(api.duration)}</span>
        </div>
        <button type="button" className="export" onClick={api.onExport}>
          Export
        </button>
      </header>
      <CommandBox layout="bar" onCommand={api.onCommand} />
      <nav className="rail" aria-label="Libraries">
        {tools.map((tool) => (
          <button
            key={tool.id}
            type="button"
            className="rail-btn"
            data-active={api.tool === tool.id ? "true" : "false"}
            onClick={() => api.onTool(tool.id)}
          >
            <Icon name={tool.id} />
            {tool.label}
          </button>
        ))}
      </nav>
      <aside className="assets">
        <AssetBrowser tool={api.tool} layout="grid" onCommand={api.onCommand} />
      </aside>
      <div className="center">
        <Preview clips={api.project.clips} time={api.time} note={api.note} />
        <Transport
          time={api.time}
          duration={api.duration}
          playing={api.playing}
          onTogglePlay={api.onTogglePlay}
          onSeek={api.onSeek}
        />
      </div>
      <Inspector clip={selected} silencesRemoved={api.project.silencesRemoved} />
      <Timeline
        tracks={api.project.tracks}
        clips={api.project.clips}
        duration={api.duration}
        time={api.time}
        selectedId={api.selectedId}
        silencesRemoved={api.project.silencesRemoved}
        onSeek={api.onSeek}
        onSelect={api.onSelect}
      />
    </div>
  )
}

function Inspector({
  clip,
  silencesRemoved,
}: {
  clip: ReturnType<typeof findClip>
  silencesRemoved: boolean
}) {
  if (!clip) {
    return (
      <aside className="inspector">
        <p className="empty">Select a clip on the timeline.</p>
      </aside>
    )
  }

  return (
    <aside className="inspector">
      <p className="eyebrow">{clip.tone === "generated" ? "AI shot" : clip.tone}</p>
      <h2>{clip.name}</h2>
      <dl>
        <div>
          <dt>Start</dt>
          <dd>{formatTime(clip.start)}</dd>
        </div>
        <div>
          <dt>End</dt>
          <dd>{formatTime(clip.start + clip.duration)}</dd>
        </div>
        <div>
          <dt>Duration</dt>
          <dd>{formatTime(clip.duration)}</dd>
        </div>
      </dl>
      {clip.tone === "title" && (
        <div className="block">
          <p className="eyebrow">Animated text</p>
          <p>Rise · Newsreader</p>
        </div>
      )}
      {clip.tone === "picture" && (
        <div className="block">
          <p className="eyebrow">Speed</p>
          <p>1.0×{silencesRemoved ? " · silences cut" : ""}</p>
        </div>
      )}
      {clip.tone === "sound" && (
        <div className="block">
          <p className="eyebrow">Volume</p>
          <div className="meter" />
        </div>
      )}
      {clip.tone === "generated" && (
        <p className="help">Inserted on the Gen track at 0:32. The vows stay underneath on Video.</p>
      )}
    </aside>
  )
}
