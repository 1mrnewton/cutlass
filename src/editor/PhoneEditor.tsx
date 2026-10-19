import { useState } from "react"
import { AssetBrowser } from "./AssetBrowser"
import { CommandBox } from "./CommandBox"
import { Icon } from "./icons"
import { scenes, tools, type EditorApi } from "./model"
import { Preview, Transport } from "./Preview"

export function PhoneEditor({ api }: { api: EditorApi }) {
  const [aiOpen, setAiOpen] = useState(false)
  const main = api.project.clips.filter((clip) => clip.trackId === "v1")
  const generated = api.project.clips.find((clip) => clip.id === "gen-shot")

  return (
    <section className="phone" aria-label="Phone editor">
      <div className="phone-nav">
        <span className="back">Projects</span>
        <strong>Amina & Jules</strong>
        <button
          type="button"
          className="ai-btn"
          aria-pressed={aiOpen}
          onClick={() => setAiOpen((open) => !open)}
        >
          AI
        </button>
      </div>
      <Preview clips={api.project.clips} time={api.time} note={api.note} />
      <Transport
        time={api.time}
        duration={api.duration}
        playing={api.playing}
        onTogglePlay={api.onTogglePlay}
        onSeek={api.onSeek}
      />
      <div className="filmstrip">
        {main.map((clip) => (
          <button
            key={clip.id}
            type="button"
            className="film"
            data-selected={api.selectedId === clip.id ? "true" : "false"}
            style={{
              flexGrow: clip.duration,
              flexBasis: 0,
              background: scenes[clip.id]?.[1] ?? "#6a7a22",
            }}
            onClick={() => {
              api.onSelect(clip.id)
              api.onSeek(clip.start)
            }}
          >
            {clip.name}
          </button>
        ))}
        <i className="film-playhead" style={{ left: `${(api.time / api.duration) * 100}%` }} />
        {generated && (
          <i className="gen-mark" style={{ left: `${(generated.start / api.duration) * 100}%` }} />
        )}
      </div>
      <div className="shelf">
        <AssetBrowser tool={api.tool} layout="row" onCommand={api.onCommand} />
      </div>
      <div className="toolrow">
        {tools.map((tool) => (
          <button
            key={tool.id}
            type="button"
            data-active={api.tool === tool.id ? "true" : "false"}
            onClick={() => api.onTool(tool.id)}
          >
            <Icon name={tool.id} />
            {tool.label}
          </button>
        ))}
      </div>
      {aiOpen && (
        <div className="sheet">
          <CommandBox layout="stack" onCommand={api.onCommand} />
        </div>
      )}
    </section>
  )
}
