import { useEffect, useLayoutEffect, useRef, useState } from "react"
import { DesktopEditor } from "./editor/DesktopEditor"
import { applyCommand, deleteSelection, formatTime, initialProject, timelineDuration, type EditorApi, type ProjectState, type ToolId } from "./editor/model"

export default function App() {
  const [project, setProject] = useState(initialProject)
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const selectedId = selectedIds.at(-1) ?? null
  const [tool, setTool] = useState<ToolId>("media")
  const [note, setNote] = useState<string | null>(null)
  const past = useRef<{ project: ProjectState; selectedIds: string[] }[]>([])
  const future = useRef<{ project: ProjectState; selectedIds: string[] }[]>([])
  const [canUndo, setCanUndo] = useState(false)
  const [canRedo, setCanRedo] = useState(false)
  const duration = timelineDuration(project.clips)
  const durationRef = useRef(duration)
  useLayoutEffect(() => {
    durationRef.current = duration
  }, [duration])

  useEffect(() => {
    if (!note) return
    const timer = window.setTimeout(() => setNote(null), 2000)
    return () => window.clearTimeout(timer)
  }, [note])

  if (project.clips.length === 0 && playing) setPlaying(false)

  useEffect(() => {
    if (!playing) return
    let frame = 0
    let last = performance.now()
    const tick = (now: number) => {
      const delta = (now - last) / 1000
      last = now
      setTime((current) => {
        const next = current + delta
        return next >= durationRef.current ? 0 : next
      })
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [playing])

  const remember = (next: ProjectState) => {
    if (next === project) return
    past.current.push({ project, selectedIds })
    if (past.current.length > 100) past.current.shift()
    future.current = []
    setCanUndo(true)
    setCanRedo(false)
  }

  const undo = () => {
    const step = past.current.pop()
    if (!step) return
    future.current.push({ project, selectedIds })
    setProject(step.project)
    setSelectedIds(step.selectedIds)
    setCanUndo(past.current.length > 0)
    setCanRedo(true)
  }

  const redo = () => {
    const step = future.current.pop()
    if (!step) return
    past.current.push({ project, selectedIds })
    setProject(step.project)
    setSelectedIds(step.selectedIds)
    setCanUndo(true)
    setCanRedo(future.current.length > 0)
  }

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault()
        if (event.shiftKey) redo()
        else undo()
        return
      }
      if (event.ctrlKey && event.key.toLowerCase() === "y") {
        event.preventDefault()
        redo()
        return
      }
      if (event.code === "Space") {
        event.preventDefault()
        if (project.clips.length === 0) return
        setPlaying((value) => !value)
        return
      }
      if ((event.key === "Backspace" || event.key === "Delete") && selectedIds.length > 0) {
        event.preventDefault()
        const result = deleteSelection(project, selectedIds)
        remember(result.state)
        setProject(result.state)
        if (result.note !== undefined) setNote(result.note)
        setSelectedIds(result.selectIds)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [project, selectedIds])

  const api: EditorApi = {
    time,
    duration,
    playing,
    project,
    selectedId,
    selectedIds,
    tool,
    note,
    onTogglePlay: () => {
      if (project.clips.length === 0) return
      setPlaying((value) => !value)
    },
    onSeek: (next) => setTime(Math.min(duration, Math.max(0, next))),
    onStill: () => {
      if (project.clips.length === 0) return
      setNote(`Still at ${formatTime(time)}`)
    },
    onSelect: (id) => setSelectedIds([id]),
    onSelectIds: setSelectedIds,
    onTool: setTool,
    onCommand: (command) => {
      const result = applyCommand(project, command)
      remember(result.state)
      setProject(result.state)
      setNote(result.note)
      if (result.selectId) setSelectedIds([result.selectId])
      if (result.seek !== undefined) setTime(result.seek)
    },
    onProject: (next, nextNote, selectId) => {
      remember(next)
      setProject(next)
      if (nextNote !== undefined) setNote(nextNote)
      if (selectId) setSelectedIds([selectId])
    },
    canUndo,
    canRedo,
    onUndo: undo,
    onRedo: redo,
  }

  return (
    <div className="app">
      <DesktopEditor api={api} />
    </div>
  )
}
