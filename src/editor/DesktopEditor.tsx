import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react"
import { AssetBrowser } from "./AssetBrowser"
import { ExportDialog } from "./ExportDialog"
import { Icon } from "./icons"
import { findClip, formatTime, frameRateLabel, insertPlacement, placementFor, projectPath, supportsDropFrame, tools, trackKindLabel, type EditorApi, type Guide, type ProjectSettings, type TrackKind, type Transition } from "./model"
import { Preview, Transport } from "./Preview"
import { ProjectDialog } from "./ProjectDialog"
import { Timeline } from "./Timeline"
import { TooltipLayer } from "./Tooltip"
import { AdjustClipInspector, AudioInspector, EffectInspector, FilterInspector, StickerInspector } from "./KindInspectors"
import { LookProvider } from "./look"
import { TextInspector } from "./TextInspector"
import { VideoInspector } from "./VideoInspector"

const TIMELINE_MIN = 160
// Head, preview padding, the 140px frame, transport, and the split.
const TIMELINE_RESERVE = 268

export function DesktopEditor({ api }: { api: EditorApi }) {
  const selected = findClip(api.project.clips, api.selectedId)
  const selectedTrack = api.project.tracks.find((track) => track.id === selected?.trackId) ?? null
  const transition = selected ? null : (api.project.transitions.find((item) => item.id === api.selectedId) ?? null)
  const guide = selected || transition ? null : (api.project.guides.find((item) => item.id === api.selectedId) ?? null)
  const desktopRef = useRef<HTMLDivElement>(null)
  const [libraryWidth, setLibraryWidth] = useState(368)
  const [inspectorWidth, setInspectorWidth] = useState(300)
  const [timelineHeight, setTimelineHeight] = useState(420)
  const [exportOpen, setExportOpen] = useState(false)
  const [projectOpen, setProjectOpen] = useState(false)

  const frame = () => {
    const box = desktopRef.current?.getBoundingClientRect()
    return { width: box?.width ?? 1280, height: box?.height ?? 800 }
  }

  useEffect(() => {
    const desktop = desktopRef.current
    if (!desktop) return
    const fit = () => setTimelineHeight((current) => Math.min(current, timelineMax(desktop.clientHeight)))
    fit()
    const observer = new ResizeObserver(fit)
    observer.observe(desktop)
    return () => observer.disconnect()
  }, [])

  return (
    <LookProvider>
    <div className="desktop" ref={desktopRef}>
      <header className="head">
        <div className="topbar">
          <div className="brand">
            <div className="traffic">
              <button type="button" className="traffic-btn" data-kind="close" aria-label="Close" />
              <button type="button" className="traffic-btn" data-kind="min" aria-label="Minimize" />
              <button type="button" className="traffic-btn" data-kind="max" aria-label="Maximize" />
            </div>
          </div>
          <div className="project-name">{api.project.settings.name}</div>
          <button type="button" className="export" onClick={() => setExportOpen(true)}>
            Export
          </button>
        </div>
      </header>
      <div className="workspace">
        <div className="library" style={{ width: libraryWidth }}>
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
            <AssetBrowser
              tool={api.tool}
              project={api.project}
              onCommand={api.onCommand}
              onNote={(note) => api.onProject(api.project, note)}
              onInsert={(id) => {
                const source = placementFor(id)
                if (!source) return
                const result = insertPlacement(api.project, source)
                api.onProject(result.state, result.note, result.selectId)
                if (result.seek !== undefined) api.onSeek(result.seek)
              }}
            />
          </aside>
        </div>
        <div
          className="split split-x"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize library"
          onPointerDown={(event) => {
            const start = libraryWidth
            beginResize(event, (dx) => {
              const max = frame().width - inspectorWidth - 360
              setLibraryWidth(clamp(start + dx, 220, max))
            })
          }}
        />
        <div className="center">
          <Preview clips={api.project.clips} time={api.time} rate={api.project.settings.frameRate} dropFrame={api.project.settings.dropFrame} />
          <Transport
            time={api.time}
            duration={api.duration}
            rate={api.project.settings.frameRate}
            dropFrame={api.project.settings.dropFrame}
            playing={api.playing}
            empty={api.project.clips.length === 0}
            onTogglePlay={api.onTogglePlay}
            onStill={api.onStill}
          />
        </div>
        <div
          className="split split-x"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize inspector"
          onPointerDown={(event) => {
            const start = inspectorWidth
            beginResize(event, (dx) => {
              const max = frame().width - libraryWidth - 360
              setInspectorWidth(clamp(start - dx, 220, max))
            })
          }}
        />
        <Inspector
          clip={selected}
          kind={selectedTrack?.kind ?? null}
          main={selectedTrack?.main ?? false}
          silencesRemoved={api.project.silencesRemoved}
          transition={transition}
          guide={guide}
          markerCount={selected ? api.project.markers.filter((marker) => marker.clipId === selected.id).length : 0}
          selectedCount={api.selectedIds.filter((id) => api.project.clips.some((clip) => clip.id === id)).length}
          settings={api.project.settings}
          onModify={() => setProjectOpen(true)}
          onNote={(note) => api.onProject(api.project, note)}
          width={inspectorWidth}
        />
      </div>
      <div
        className="split split-y"
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize timeline"
        onPointerDown={(event) => {
          const start = timelineHeight
          beginResize(event, (_dx, dy) => {
            const max = timelineMax(frame().height)
            setTimelineHeight(clamp(start - dy, TIMELINE_MIN, max))
          })
        }}
      />
      <TooltipLayer />
      <Timeline
        project={api.project}
        duration={api.duration}
        time={api.time}
        selectedIds={api.selectedIds}
        onSeek={api.onSeek}
        onSelect={api.onSelect}
        onSelectIds={api.onSelectIds}
        onProject={api.onProject}
        canUndo={api.canUndo}
        canRedo={api.canRedo}
        onUndo={api.onUndo}
        onRedo={api.onRedo}
        note={api.note}
        height={timelineHeight}
      />
      {exportOpen && (
        <ExportDialog duration={api.duration} name={api.project.settings.name} onClose={() => setExportOpen(false)} />
      )}
      {projectOpen && (
        <ProjectDialog
          settings={api.project.settings}
          onClose={() => setProjectOpen(false)}
          onSave={(settings) => {
            api.onProject({ ...api.project, settings })
            setProjectOpen(false)
          }}
        />
      )}
    </div>
    </LookProvider>
  )
}

function timelineMax(height: number) {
  return Math.max(TIMELINE_MIN, height - TIMELINE_RESERVE)
}

function clamp(value: number, min: number, max: number) {
  const upper = Math.max(min, max)
  return Math.min(upper, Math.max(min, value))
}

function beginResize(
  event: ReactPointerEvent<HTMLElement>,
  apply: (dx: number, dy: number) => void,
) {
  if (event.button !== 0) return
  event.preventDefault()
  const handle = event.currentTarget
  const originX = event.clientX
  const originY = event.clientY
  try {
    handle.setPointerCapture(event.pointerId)
  } catch {
    // Listening on the handle still tracks the drag when capture is unavailable.
  }
  handle.dataset.active = "true"
  const previousCursor = document.body.style.cursor
  const previousSelect = document.body.style.userSelect
  document.body.style.cursor = handle.classList.contains("split-y") ? "ns-resize" : "ew-resize"
  document.body.style.userSelect = "none"

  const move = (pointer: PointerEvent) => {
    apply(pointer.clientX - originX, pointer.clientY - originY)
  }
  const end = () => {
    handle.dataset.active = "false"
    document.body.style.cursor = previousCursor
    document.body.style.userSelect = previousSelect
    handle.removeEventListener("pointermove", move)
    handle.removeEventListener("pointerup", end)
    handle.removeEventListener("pointercancel", end)
  }
  handle.addEventListener("pointermove", move)
  handle.addEventListener("pointerup", end)
  handle.addEventListener("pointercancel", end)
}

function Inspector({
  clip,
  kind,
  main,
  silencesRemoved,
  transition,
  guide,
  markerCount,
  selectedCount,
  settings,
  onModify,
  onNote,
  width,
}: {
  clip: ReturnType<typeof findClip>
  kind: TrackKind | null
  main: boolean
  silencesRemoved: boolean
  transition: Transition | null
  guide: Guide | null
  markerCount: number
  selectedCount: number
  settings: ProjectSettings
  onModify: () => void
  onNote: (note: string) => void
  width: number
}) {
  if (!clip && guide) {
    return (
      <aside className="inspector" style={{ width }}>
        <p className="eyebrow">Reference line</p>
        <h2>{guide.axis === "time" ? formatTime(guide.at) : "Across the tracks"}</h2>
        <p className="help">Pulled off the ruler. Delete removes it. Lock keeps every line where it is.</p>
      </aside>
    )
  }

  if (!clip && transition) {
    return (
      <aside className="inspector" style={{ width }}>
        <p className="eyebrow">Transition</p>
        <h2>{transition.name}</h2>
        <dl>
          <div>
            <dt>Duration</dt>
            <dd>{formatTime(transition.duration)}</dd>
          </div>
        </dl>
        <p className="help">Sits on a cut in the main track. Split cannot cut through it.</p>
      </aside>
    )
  }

  if (!clip) {
    const importLabel = settings.importMaterial === "keep" ? "Keep in original place" : "Copy to project"
    return (
      <aside className="inspector project-inspector" style={{ width }}>
        <div className="project-details">
          <h2 className="panel-title">Details</h2>
          <dl className="project-facts">
            <div>
              <dt>Name</dt>
              <dd>{settings.name}</dd>
            </div>
            <div>
              <dt>Saved</dt>
              <dd>{projectPath(settings.name)}</dd>
            </div>
            <div>
              <dt>Color space</dt>
              <dd>{settings.colorSpace}</dd>
            </div>
            <div>
              <dt>Import material</dt>
              <dd>{importLabel}</dd>
            </div>
            <div>
              <dt>Free layer</dt>
              <dd>{settings.freeLayer ? "Turned on" : "Turned off"}</dd>
            </div>
            <div>
              <dt>Proxy</dt>
              <dd>{settings.proxy ? "Turned on" : "Turned off"}</dd>
            </div>
          </dl>
          <dl className="project-facts project-timeline">
            <div>
              <dt>Ratio</dt>
              <dd>{settings.ratio}</dd>
            </div>
            <div>
              <dt>Resolution</dt>
              <dd>{settings.resolution}</dd>
            </div>
            <div>
              <dt>Frame rate</dt>
              <dd>{frameRateLabel(settings.frameRate).replace(" ", "")}</dd>
            </div>
            {supportsDropFrame(settings.frameRate) && (
              <div>
                <dt>Timecode</dt>
                <dd>{settings.dropFrame ? "Drop frame" : "Non-drop"}</dd>
              </div>
            )}
          </dl>
        </div>
        <footer className="project-foot">
          <button type="button" className="modify" onClick={onModify}>
            Modify
          </button>
        </footer>
      </aside>
    )
  }

  if (kind === "video") {
    return <VideoInspector key={clip.id} clip={clip} selectedCount={selectedCount} width={width} onNote={onNote} />
  }

  if (kind === "text" || kind === "text_to_video_text") {
    return <TextInspector key={clip.id} clip={clip} selectedCount={selectedCount} width={width} onNote={onNote} />
  }

  if (kind === "audio" || kind === "text_to_video_text_audio") {
    return <AudioInspector key={clip.id} clip={clip} width={width} onNote={onNote} />
  }

  if (kind === "sticker") return <StickerInspector key={clip.id} clip={clip} width={width} />
  if (kind === "effect" || kind === "plugin_effect") return <EffectInspector key={clip.id} clip={clip} width={width} />
  if (kind === "filter") return <FilterInspector key={clip.id} clip={clip} width={width} />
  if (kind === "adjust") return <AdjustClipInspector key={clip.id} clip={clip} width={width} />

  return (
    <aside className="inspector" style={{ width }}>
      <p className="eyebrow">{main ? "Main" : kind ? trackKindLabel[kind] : clip.tone}</p>
      <h2>{clip.name}</h2>
      {selectedCount > 1 && <p className="help">{selectedCount} clips selected.</p>}
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
        <div>
          <dt>Trim in</dt>
          <dd>{formatTime(clip.trimIn)}</dd>
        </div>
        <div>
          <dt>Trim out</dt>
          <dd>{formatTime(Math.max(0, clip.source - clip.trimIn - clip.duration))}</dd>
        </div>
      </dl>
      {markerCount > 0 && (
        <p className="help">{markerCount === 1 ? "1 marker on this clip." : `${markerCount} markers on this clip.`}</p>
      )}
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
        <p className="help">Inserted on a video track at 0:32. The vows stay on the main track underneath.</p>
      )}
      {clip.tone === "sticker" && (
        <p className="help">A sticker track. It is not part of the video.</p>
      )}
      {clip.tone === "effect" && (
        <p className="help">An effect track over the vows.</p>
      )}
      {clip.tone === "filter" && (
        <p className="help">A filter track over the portraits.</p>
      )}
      {clip.tone === "adjust" && (
        <p className="help">An adjustment track across the whole film.</p>
      )}
    </aside>
  )
}
