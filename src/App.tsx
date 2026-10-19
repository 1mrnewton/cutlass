import { useEffect, useState } from "react"
import { DesktopEditor } from "./editor/DesktopEditor"
import { PhoneEditor } from "./editor/PhoneEditor"
import { DURATION, applyCommand, initialProject, type EditorApi, type ToolId } from "./editor/model"

type Shell = "desktop" | "phone"

export default function App() {
  const [shell, setShell] = useState<Shell>("desktop")
  const [project, setProject] = useState(initialProject)
  const [time, setTime] = useState(32)
  const [playing, setPlaying] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>("vows")
  const [tool, setTool] = useState<ToolId>("media")
  const [note, setNote] = useState<string | null>(null)

  useEffect(() => {
    if (!playing) return
    let frame = 0
    let last = performance.now()
    const tick = (now: number) => {
      const delta = (now - last) / 1000
      last = now
      setTime((current) => {
        const next = current + delta
        return next >= DURATION ? 0 : next
      })
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [playing])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return
      if (event.code !== "Space") return
      event.preventDefault()
      setPlaying((value) => !value)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  const api: EditorApi = {
    time,
    duration: DURATION,
    playing,
    project,
    selectedId,
    tool,
    note,
    onTogglePlay: () => setPlaying((value) => !value),
    onSeek: (next) => setTime(Math.min(DURATION, Math.max(0, next))),
    onSelect: setSelectedId,
    onTool: setTool,
    onCommand: (command) => {
      const result = applyCommand(project, command)
      setProject(result.state)
      setNote(result.note)
      if (result.selectId) setSelectedId(result.selectId)
      if (result.seek !== undefined) setTime(result.seek)
    },
    onExport: () => setNote("Export isn't part of this prototype."),
  }

  return (
    <div className="app">
      <header className="proto">
        <span className="proto-mark">Cutlass rewrite</span>
        <span>Prototype</span>
        <div className="seg" role="group" aria-label="Editor shell">
          <button type="button" data-active={shell === "desktop" ? "true" : "false"} onClick={() => setShell("desktop")}>
            Desktop
          </button>
          <button type="button" data-active={shell === "phone" ? "true" : "false"} onClick={() => setShell("phone")}>
            Phone
          </button>
        </div>
        <span className="hint">Space plays and pauses</span>
      </header>
      <div className="stage" data-shell={shell}>
        {shell === "desktop" ? (
          <DesktopEditor api={api} />
        ) : (
          <div className="phone-stage">
            <PhoneEditor api={api} />
          </div>
        )}
      </div>
    </div>
  )
}
