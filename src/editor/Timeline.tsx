import { useEffect, useLayoutEffect, useRef, useState, type ComponentProps, type DragEvent as ReactDragEvent, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent } from "react"
import { ClipMenu, menuFor } from "./ClipMenu"
import { Icon } from "./icons"
import { Tip } from "./Tooltip"
import {
  addGuide,
  addMarker,
  addPlacement,
  addTransition,
  adsorbTime,
  assetDragId,
  deleteSelection,
  endAssetDrag,
  moveClip,
  moveGuide,
  nearestCut,
  placementFor,
  planAdd,
  planMove,
  removeMarker,
  frameRateFps,
  frameSeconds,
  formatTime,
  rulerTicks,
  zoomCeiling,
  setGuidesLocked,
  setGuidesVisible,
  setLinkage,
  setMagnet,
  setTrackLocked,
  splitClip,
  trackKindLabel,
  trimClip,
  type Clip,
  type DropPlan,
  type Guide,
  type ProjectState,
  type Tone,
  type Track,
  type TrackKind,
} from "./model"

const trackIcon: Record<TrackKind, ComponentProps<typeof Icon>["name"]> = {
  video: "video",
  audio: "audio",
  sticker: "stickers",
  text: "text",
  effect: "effects",
  filter: "filters",
  adjust: "adjustments",
  text_to_video_text: "textVideo",
  text_to_video_text_audio: "audio",
  composition: "composition",
  plugin_effect: "plugin",
  adcube: "ad",
  mixed: "mixed",
}

const videoKinds: TrackKind[] = ["video"]
const timedKinds: TrackKind[] = ["video", "audio", "text_to_video_text_audio"]
const pictureKinds: TrackKind[] = ["video", "sticker", "text", "text_to_video_text", "composition", "mixed"]
const framedKinds: TrackKind[] = ["video", "composition", "mixed"]

const MIN_WIDTH_RATIO = 0.25
const FIT_WIDTH_RATIO = 0.75
const EMPTY_WIDTH_RATIO = 0.33
function zoomFloor(lane: number, seconds: number, ceiling: number) {
  if (lane <= 0 || seconds <= 0) return 0
  return Math.min(ceiling, (lane * MIN_WIDTH_RATIO) / seconds)
}

function clampPixels(pps: number, lane: number, seconds: number, ceiling: number) {
  return Math.min(ceiling, Math.max(zoomFloor(lane, seconds, ceiling), pps))
}

type TimelineTool = "select" | "split" | "marker" | "beat"

type Hover = {
  name: string
  tone: Tone
  kind: TrackKind
  start: number
  duration: number
  plan: DropPlan
  movingId?: string
}

function timeAt(sheet: HTMLDivElement, clientX: number, duration: number) {
  const rect = sheet.getBoundingClientRect()
  const ratio = (clientX - rect.left) / Math.max(1, rect.width)
  return Math.min(duration, Math.max(0, ratio * duration))
}

function trackAt(sheet: HTMLDivElement, clientY: number, tracks: Track[], layout: StackLayout) {
  const rect = sheet.getBoundingClientRect()
  let y = clientY - rect.top - layout.stackTop
  if (y < 0) return null
  for (let index = 0; index < tracks.length; index += 1) {
    const track = tracks[index]
    const hit = laneHeight(track) + (index < tracks.length - 1 ? layout.gap : 0)
    if (y < hit) return track.id
    y -= hit
  }
  return null
}

function snapThreshold(sheet: HTMLDivElement, duration: number, snap: boolean) {
  if (!snap) return 0
  const width = sheet.getBoundingClientRect().width
  return width > 0 ? (10 / width) * duration : 0
}

function laneOf(tracks: Track[], trackId: string) {
  return tracks.find((track) => track.id === trackId)
}

const TOAST_DROP_MS = 320

function TimelineToast({ note }: { note: string | null }) {
  const ref = useRef<HTMLDivElement>(null)
  const [text, setText] = useState<string | null>(null)
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches
  if (note !== null && note !== text) setText(note)
  if (note === null && reduce && text !== null) setText(null)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el || !text || reduce) return
    const dy = toastDrop(el)
    const anim = el.animate(note ? toastFrames(dy) : toastFade(), {
      duration: note ? TOAST_DROP_MS : 160,
      easing: "linear",
      fill: "both",
    })
    if (!note) anim.onfinish = () => setText(null)
    return () => {
      anim.onfinish = null
      anim.cancel()
    }
  }, [note, text, reduce])

  if (!text) return null
  return (
    <div className="toast" role="status" ref={ref}>
      {text}
    </div>
  )
}

function toastDrop(el: HTMLElement) {
  const toolbar = document.querySelector(".tl-toolbar")
  if (!toolbar) return -48
  const toast = el.getBoundingClientRect()
  const bar = toolbar.getBoundingClientRect()
  return bar.top + bar.height / 2 - (toast.top + toast.height / 2)
}

function toastFrames(dy: number): Keyframe[] {
  const at = (factor: number) => `translate(-50%, calc(-50% + ${dy * factor}px))`
  return [
    { transform: at(1), opacity: 0, offset: 0 },
    { transform: at(0.9), opacity: 1, offset: 0.12 },
    { transform: at(0.22), opacity: 1, offset: 0.55 },
    { transform: at(0), opacity: 1, offset: 1 },
  ]
}

function toastFade(): Keyframe[] {
  const rest = "translate(-50%, -50%)"
  return [
    { transform: rest, opacity: 1 },
    { transform: rest, opacity: 0 },
  ]
}

export function Timeline({
  project,
  duration,
  time,
  selectedIds,
  onSeek,
  onSelect,
  onSelectIds,
  onProject,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  note,
  height,
}: {
  project: ProjectState
  duration: number
  time: number
  selectedIds: string[]
  onSeek: (time: number) => void
  onSelect: (id: string) => void
  onSelectIds: (ids: string[]) => void
  onProject: (project: ProjectState, note?: string | null, selectId?: string) => void
  canUndo: boolean
  canRedo: boolean
  onUndo: () => void
  onRedo: () => void
  note: string | null
  height: number
}) {
  const [tool, setTool] = useState<TimelineTool>("select")
  const [snap, setSnap] = useState(true)
  const [pixels, setPixels] = useState(0)
  const [hover, setHover] = useState<Hover | null>(null)
  const [preview, setPreview] = useState<ProjectState | null>(null)
  const [drag, setDrag] = useState<{
    id: string
    name: string
    tone: Tone
    kind: TrackKind
    main: boolean
    duration: number
    start: number
    top: number
  } | null>(null)
  const [cutAt, setCutAt] = useState<number | null>(null)
  const [adsorb, setAdsorb] = useState<number | null>(null)
  const [draftGuide, setDraftGuide] = useState<Pick<Guide, "axis" | "at"> | null>(null)
  const [guideDrag, setGuideDrag] = useState<{ id: string; at: number } | null>(null)
  const [laneWidth, setLaneWidth] = useState(0)
  const [menu, setMenu] = useState<{ id: string; x: number; y: number } | null>(null)
  const [area, setArea] = useState(0)
  const [marquee, setMarquee] = useState<{ left: number; top: number; width: number; height: number } | null>(null)
  const rootRef = useRef<HTMLElement>(null)
  const lanesRef = useRef<HTMLDivElement>(null)
  const headRef = useRef<HTMLDivElement>(null)
  const rulerRef = useRef<HTMLDivElement>(null)
  const sheetRef = useRef<HTMLDivElement>(null)
  const playheadRef = useRef<HTMLDivElement>(null)
  const playheadFrameRef = useRef<HTMLDivElement>(null)
  const movedRef = useRef(false)
  const hoverRef = useRef<Hover | null>(null)
  const projectRef = useRef(project)
  const liveRef = useRef({ time, duration, snap, onProject, onSelect, onSelectIds, onSeek, tool, selectedIds })
  const pixelsRef = useRef(pixels)
  const zoomAnchor = useRef<number | null>(null)
  const tuned = useRef(false)
  const hadClips = useRef(false)
  useLayoutEffect(() => {
    projectRef.current = project
    pixelsRef.current = pixels
    liveRef.current = { time, duration, snap, onProject, onSelect, onSelectIds, onSeek, tool, selectedIds }
  })

  useLayoutEffect(() => {
    const lanes = lanesRef.current
    if (!lanes) return
    const measure = () => {
      setLaneWidth(lanes.clientWidth)
      setArea(lanes.clientHeight)
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(lanes)
    return () => observer.disconnect()
  }, [])

  const trackKey = project.tracks.map((track) => `${track.id}:${track.main ? 1 : 0}`).join("|")
  useLayoutEffect(() => {
    const lanes = lanesRef.current
    if (!lanes || area <= 0) return
    const next = stackLayout(projectRef.current.tracks, area).scrollTop
    if (Math.abs(lanes.scrollTop - next) > 1) lanes.scrollTop = next
    if (headRef.current) headRef.current.scrollTop = lanes.scrollTop
  }, [area, trackKey])

  const contentWidth = laneWidth > 0 && pixels > 0 ? Math.max(1, pixels * duration) : 0
  const rulerWidth = laneWidth > 0 ? Math.max(laneWidth, contentWidth) : 0
  const rulerDuration = contentWidth > 0 ? duration * (rulerWidth / contentWidth) : duration

  const syncScroll = () => {
    const lanes = lanesRef.current
    if (!lanes) return
    if (headRef.current) headRef.current.scrollTop = lanes.scrollTop
    if (rulerRef.current) rulerRef.current.style.transform = `translateX(${-lanes.scrollLeft}px)`
  }

  const clipCount = project.clips.length
  const empty = clipCount === 0
  const fps = frameRateFps(project.settings.frameRate)
  const ceiling = zoomCeiling(fps)
  if (empty && tool !== "select") setTool("select")
  useLayoutEffect(() => {
    if (laneWidth <= 0 || duration <= 0) return
    const hasClips = clipCount > 0
    let next = pixelsRef.current
    if (!tuned.current && !hasClips) next = (laneWidth * EMPTY_WIDTH_RATIO) / duration
    else if (!tuned.current && hasClips && !hadClips.current) {
      next = (laneWidth * FIT_WIDTH_RATIO) / duration
      tuned.current = true
    } else next = clampPixels(next, laneWidth, duration, ceiling)
    hadClips.current = hasClips
    if (next > 0 && Math.abs(next - pixelsRef.current) < 0.01) return
    pixelsRef.current = next
    setPixels(next)
  }, [laneWidth, duration, clipCount, ceiling])

  const applyZoom = (next: number, pointer?: number) => {
    if (projectRef.current.clips.length === 0) return
    const lanes = lanesRef.current
    const seconds = liveRef.current.duration
    const lane = lanes?.clientWidth ?? 0
    const current = pixelsRef.current
    const clamped = clampPixels(next, lane, seconds, zoomCeiling(frameRateFps(projectRef.current.settings.frameRate)))
    if (!lanes || current <= 0 || Math.abs(clamped - current) < 0.01) return
    tuned.current = true
    const at = pointer ?? lane / 2
    const span = Math.max(1, current * seconds)
    const ratio = (lanes.scrollLeft + at) / span
    zoomAnchor.current = Math.max(0, ratio * clamped * seconds - at)
    pixelsRef.current = clamped
    setPixels(clamped)
  }

  useLayoutEffect(() => {
    const lanes = lanesRef.current
    const anchor = zoomAnchor.current
    if (!lanes || anchor == null) return
    zoomAnchor.current = null
    lanes.scrollLeft = anchor
    syncScroll()
  }, [pixels])

  useEffect(() => {
    const root = rootRef.current
    if (!root) return
    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey) return
      event.preventDefault()
      const lanes = lanesRef.current
      if (!lanes) return
      const delta = Math.max(-80, Math.min(80, event.deltaY))
      const box = lanes.getBoundingClientRect()
      const pointer = Math.min(box.width, Math.max(0, event.clientX - box.left))
      applyZoom(pixelsRef.current * Math.exp(-delta * 0.01), pointer)
    }
    root.addEventListener("wheel", onWheel, { passive: false })
    return () => root.removeEventListener("wheel", onWheel)
  }, [])

  useLayoutEffect(() => {
    const frame = playheadFrameRef.current
    const sheet = sheetRef.current
    const playhead = playheadRef.current
    if (!frame || !sheet || !playhead) return
    const place = () => {
      const frameBox = frame.getBoundingClientRect()
      const sheetBox = sheet.getBoundingClientRect()
      const at = frameSeconds(time, fps)
      const x = sheetBox.left - frameBox.left + (at / duration) * sheetBox.width
      playhead.style.left = `${x}px`
    }
    place()
    const observer = new ResizeObserver(place)
    observer.observe(frame)
    observer.observe(sheet)
    const scroller = sheet.parentElement
    scroller?.addEventListener("scroll", place, { passive: true })
    return () => {
      observer.disconnect()
      scroller?.removeEventListener("scroll", place)
    }
  }, [time, duration, contentWidth, fps])

  const readPointer = (clientX: number, clientY: number) => {
    const sheet = sheetRef.current
    const live = liveRef.current
    if (!sheet) return null
    return {
      sheet,
      at: timeAt(sheet, clientX, live.duration),
      trackId: trackAt(sheet, clientY, projectRef.current.tracks, stackLayout(projectRef.current.tracks, lanesRef.current?.clientHeight ?? area)),
      threshold: snapThreshold(sheet, live.duration, live.snap),
    }
  }

  const nudgeScroll = (clientX: number) => {
    const lanes = lanesRef.current
    if (!lanes) return
    const box = lanes.getBoundingClientRect()
    if (clientX < box.left + 140) lanes.scrollLeft -= 16
    else if (clientX > box.right - 28) lanes.scrollLeft += 16
    syncScroll()
  }

  const rowAt = (clientY: number) => {
    const sheet = sheetRef.current
    if (!sheet) return 0
    const box = sheet.getBoundingClientRect()
    const layout = stackLayout(projectRef.current.tracks, lanesRef.current?.clientHeight ?? area)
    return Math.min(1, Math.max(0, (clientY - box.top - layout.stackTop) / Math.max(1, layout.stackHeight)))
  }

  const trackLocked = (clip: Clip) => {
    const track = projectRef.current.tracks.find((item) => item.id === clip.trackId)
    if (!track?.locked) return false
    liveRef.current.onProject(projectRef.current, "This track is locked.")
    return true
  }

  const markClip = (clip: Clip, at: number, beat: boolean) => {
    const result = addMarker(projectRef.current, at, clip.id, beat)
    liveRef.current.onProject(result.state, result.note ?? null, clip.id)
  }

  const cutClip = (clip: Clip, at: number) => {
    const result = splitClip(projectRef.current, clip.id, at)
    if (result.note) liveRef.current.onProject(result.state, result.note, result.selectId)
    else if (result.selectId) liveRef.current.onProject(result.state, null, result.selectId)
  }

  const onDragOver = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!assetDragId()) return
    event.preventDefault()
    event.dataTransfer.dropEffect = "copy"
    nudgeScroll(event.clientX)
    const sourceId = assetDragId()
    const source = sourceId ? placementFor(sourceId) : null
    const pointer = readPointer(event.clientX, event.clientY)
    if (!source || !pointer) {
      setHover(null)
      setCutAt(null)
      setAdsorb(null)
      return
    }
    if (!source.kind) {
      setHover(null)
      setAdsorb(null)
      setCutAt(nearestCut(projectRef.current, pointer.at)?.at ?? null)
      return
    }
    setCutAt(null)
    const plan = planAdd(projectRef.current, source, pointer.at, pointer.trackId, liveRef.current.time, pointer.threshold, liveRef.current.duration)
    setAdsorb(plan.guide)
    setHover({ name: source.name, tone: source.tone, kind: source.kind, start: plan.start, duration: source.duration, plan })
  }

  const onDrop = (event: ReactDragEvent<HTMLDivElement>) => {
    event.preventDefault()
    const sourceId = assetDragId()
    const source = sourceId ? placementFor(sourceId) : null
    endAssetDrag()
    setHover(null)
    setCutAt(null)
    setAdsorb(null)
    if (!source) return
    const pointer = readPointer(event.clientX, event.clientY)
    if (!pointer) return
    if (!source.kind) {
      const result = addTransition(projectRef.current, source.name, source.duration, pointer.at)
      liveRef.current.onProject(result.state, result.note, result.selectId)
      return
    }
    const plan = planAdd(projectRef.current, source, pointer.at, pointer.trackId, liveRef.current.time, pointer.threshold, liveRef.current.duration)
    if (!plan.ok) {
      liveRef.current.onProject(projectRef.current, plan.note ?? null)
      return
    }
    const result = addPlacement(projectRef.current, source, plan)
    liveRef.current.onProject(result.state, `Added ${source.name}.`, result.selectId)
  }

  const onTrimPointerDown = (event: ReactPointerEvent<HTMLSpanElement>, clip: Clip, edge: "in" | "out") => {
    event.stopPropagation()
    if (trackLocked(clip)) return
    const pointer = readPointer(event.clientX, event.clientY)
    if (!pointer) return
    if (liveRef.current.tool === "split") {
      cutClip(clip, pointer.at)
      return
    }
    if (liveRef.current.tool === "marker" || liveRef.current.tool === "beat") {
      markClip(clip, pointer.at, liveRef.current.tool === "beat")
      return
    }
    const originX = event.clientX
    let moving = false
    const target = event.currentTarget
    try {
      target.setPointerCapture(event.pointerId)
    } catch {
      // Pointer capture can fail; the window listeners still follow the drag.
    }
    const move = (pointerEvent: PointerEvent) => {
      if (!moving && Math.abs(pointerEvent.clientX - originX) < 4) return
      moving = true
      movedRef.current = true
      nudgeScroll(pointerEvent.clientX)
      const next = readPointer(pointerEvent.clientX, pointerEvent.clientY)
      if (!next) return
      const live = liveRef.current
      setAdsorb(adsorbTime(projectRef.current, next.at, 0, live.time, live.duration, next.threshold, clip.id))
      setPreview(trimClip(projectRef.current, clip.id, edge, next.at, live.time, next.threshold, live.duration))
    }
    const up = (pointerEvent: PointerEvent) => {
      target.removeEventListener("pointermove", move)
      target.removeEventListener("pointerup", up)
      target.removeEventListener("pointercancel", up)
      setPreview(null)
      setAdsorb(null)
      if (!moving) {
        chooseClip(clip.id, pointerEvent.shiftKey)
        return
      }
      const next = readPointer(pointerEvent.clientX, pointerEvent.clientY)
      if (!next) return
      const live = liveRef.current
      live.onProject(trimClip(projectRef.current, clip.id, edge, next.at, live.time, next.threshold, live.duration), null, clip.id)
    }
    target.addEventListener("pointermove", move)
    target.addEventListener("pointerup", up)
    target.addEventListener("pointercancel", up)
  }

  const onClipPointerDown = (event: ReactPointerEvent<HTMLButtonElement>, clip: Clip) => {
    event.stopPropagation()
    if (event.button !== 0 || event.ctrlKey) return
    if (trackLocked(clip)) return
    const pointer = readPointer(event.clientX, event.clientY)
    if (!pointer) return
    if (liveRef.current.tool === "split") {
      cutClip(clip, pointer.at)
      return
    }
    if (liveRef.current.tool === "marker" || liveRef.current.tool === "beat") {
      markClip(clip, pointer.at, liveRef.current.tool === "beat")
      return
    }
    const originX = event.clientX
    const originY = event.clientY
    const grabOffset = pointer.at - clip.start
    const grabY = event.clientY - event.currentTarget.getBoundingClientRect().top
    const from = laneOf(projectRef.current.tracks, clip.trackId)
    let moving = false
    const target = event.currentTarget
    try {
      target.setPointerCapture(event.pointerId)
    } catch {
      // Pointer capture can fail; the window listeners still follow the drag.
    }
    const move = (pointerEvent: PointerEvent) => {
      if (!moving && Math.hypot(pointerEvent.clientX - originX, pointerEvent.clientY - originY) < 4) return
      moving = true
      movedRef.current = true
      nudgeScroll(pointerEvent.clientX)
      const next = readPointer(pointerEvent.clientX, pointerEvent.clientY)
      if (!next) return
      const live = liveRef.current
      const plan = planMove(projectRef.current, clip.id, next.at - grabOffset, next.trackId, live.time, next.threshold, live.duration)
      setAdsorb(plan.guide)
      setPreview(null)
      const sheet = sheetRef.current
      const top = sheet ? pointerEvent.clientY - sheet.getBoundingClientRect().top - grabY : 0
      setDrag({
        id: clip.id,
        name: clip.name,
        tone: clip.tone,
        kind: from?.kind ?? "video",
        main: Boolean(from?.main),
        duration: clip.duration,
        start: Math.max(0, next.at - grabOffset),
        top,
      })
      const nextHover: Hover = {
        name: clip.name,
        tone: clip.tone,
        kind: from?.kind ?? "video",
        start: plan.start,
        duration: clip.duration,
        plan,
        movingId: clip.id,
      }
      hoverRef.current = nextHover
      setHover(nextHover)
    }
    const up = (pointerEvent: PointerEvent) => {
      target.removeEventListener("pointermove", move)
      target.removeEventListener("pointerup", up)
      target.removeEventListener("pointercancel", up)
      hoverRef.current = null
      setHover(null)
      setPreview(null)
      setDrag(null)
      setAdsorb(null)
      if (!moving) {
        chooseClip(clip.id, pointerEvent.shiftKey)
        return
      }
      const next = readPointer(pointerEvent.clientX, pointerEvent.clientY)
      if (!next) return
      const live = liveRef.current
      const plan = planMove(projectRef.current, clip.id, next.at - grabOffset, next.trackId, live.time, next.threshold, live.duration)
      if (!plan.ok) {
        liveRef.current.onProject(projectRef.current, plan.note ?? null)
        return
      }
      liveRef.current.onProject(moveClip(projectRef.current, clip.id, plan), null, clip.id)
    }
    target.addEventListener("pointermove", move)
    target.addEventListener("pointerup", up)
    target.addEventListener("pointercancel", up)
  }

  const onClipMenu = (event: ReactMouseEvent<HTMLButtonElement>, clip: Clip) => {
    event.preventDefault()
    event.stopPropagation()
    if (!selectedIds.includes(clip.id)) onSelect(clip.id)
    setMenu({ id: clip.id, x: event.clientX, y: event.clientY })
  }

  const onMenuPick = (id: string, label: string) => {
    const clip = projectRef.current.clips.find((item) => item.id === menu?.id)
    setMenu(null)
    if (!clip) return
    const live = liveRef.current
    const ids = live.selectedIds.includes(clip.id) ? live.selectedIds : [clip.id]
    if (id === "delete" || id === "cut") {
      const result = deleteSelection(projectRef.current, ids)
      const note = result.note ?? (id === "cut" && result.selectIds.length < ids.length ? (ids.length > 1 ? `Cut ${ids.length} clips` : `Cut ${clip.name}`) : undefined)
      live.onProject(result.state, note)
      live.onSelectIds(result.selectIds)
      return
    }
    if (id === "copy") {
      live.onProject(projectRef.current, ids.length > 1 ? `Copied ${ids.length} clips` : `Copied ${clip.name}`)
      return
    }
    if (id === "export" || id === "render-preview" || id === "render-full") {
      live.onProject(projectRef.current, "The prototype does not write a file.")
      return
    }
    if (id === "open-file") {
      live.onProject(projectRef.current, "Library cards are placeholders.")
      return
    }
    live.onProject(projectRef.current, label)
  }

  const pullGuide = (event: ReactPointerEvent<HTMLDivElement>, axis: Guide["axis"]) => {
    if (projectRef.current.guidesLocked || !projectRef.current.guidesVisible) return
    event.stopPropagation()
    const originX = event.clientX
    const originY = event.clientY
    let moving = false
    let at = axis === "time" ? timeAt(event.currentTarget, event.clientX, liveRef.current.duration) : 0
    const target = event.currentTarget
    try {
      target.setPointerCapture(event.pointerId)
    } catch {
      // Pointer capture can fail; the window listeners still follow the drag.
    }
    const move = (pointerEvent: PointerEvent) => {
      if (!moving && Math.hypot(pointerEvent.clientX - originX, pointerEvent.clientY - originY) < 4) return
      moving = true
      movedRef.current = true
      nudgeScroll(pointerEvent.clientX)
      at = axis === "time" ? (readPointer(pointerEvent.clientX, pointerEvent.clientY)?.at ?? at) : rowAt(pointerEvent.clientY)
      setDraftGuide({ axis, at })
    }
    const up = (pointerEvent: PointerEvent) => {
      target.removeEventListener("pointermove", move)
      target.removeEventListener("pointerup", up)
      target.removeEventListener("pointercancel", up)
      setDraftGuide(null)
      if (!moving) {
        if (axis === "time" && projectRef.current.clips.length > 0) {
          const next = readPointer(pointerEvent.clientX, pointerEvent.clientY)
          if (next) liveRef.current.onSeek(next.at)
        }
        return
      }
      const result = addGuide(projectRef.current, axis, at)
      liveRef.current.onProject(result.state, result.note ?? null, result.selectId)
    }
    target.addEventListener("pointermove", move)
    target.addEventListener("pointerup", up)
    target.addEventListener("pointercancel", up)
  }

  const onGuidePointerDown = (event: ReactPointerEvent<HTMLDivElement>, guide: Guide) => {
    event.stopPropagation()
    const originX = event.clientX
    const originY = event.clientY
    let moving = false
    let at = guide.at
    const target = event.currentTarget
    try {
      target.setPointerCapture(event.pointerId)
    } catch {
      // Pointer capture can fail; the window listeners still follow the drag.
    }
    const move = (pointerEvent: PointerEvent) => {
      if (projectRef.current.guidesLocked) return
      if (!moving && Math.hypot(pointerEvent.clientX - originX, pointerEvent.clientY - originY) < 4) return
      moving = true
      movedRef.current = true
      nudgeScroll(pointerEvent.clientX)
      at = guide.axis === "time" ? (readPointer(pointerEvent.clientX, pointerEvent.clientY)?.at ?? at) : rowAt(pointerEvent.clientY)
      setGuideDrag({ id: guide.id, at })
    }
    const up = () => {
      target.removeEventListener("pointermove", move)
      target.removeEventListener("pointerup", up)
      target.removeEventListener("pointercancel", up)
      setGuideDrag(null)
      if (!moving) {
        liveRef.current.onSelect(guide.id)
        return
      }
      liveRef.current.onProject(moveGuide(projectRef.current, guide.id, at), null, guide.id)
    }
    target.addEventListener("pointermove", move)
    target.addEventListener("pointerup", up)
    target.addEventListener("pointercancel", up)
  }

  const chooseClip = (clipId: string, shift: boolean) => {
    const live = liveRef.current
    if (!shift) {
      live.onSelect(clipId)
      return
    }
    const clipIds = new Set(projectRef.current.clips.map((clip) => clip.id))
    const current = live.selectedIds.filter((id) => clipIds.has(id))
    const next = current.includes(clipId) ? current.filter((id) => id !== clipId) : [...current, clipId]
    live.onSelectIds(next)
  }

  const onSheetPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || liveRef.current.tool !== "select") return
    const target = event.target
    if (!(target instanceof Element)) return
    if (target.closest(".clip, .transition, .guide-line, .ruler, .marker, .beat")) return
    const sheet = event.currentTarget
    const originX = event.clientX
    const originY = event.clientY
    const shift = event.shiftKey
    let moving = false
    const planeBox = () => sheetRef.current?.getBoundingClientRect() ?? sheet.getBoundingClientRect()
    try {
      sheet.setPointerCapture(event.pointerId)
    } catch {
      // Pointer capture can fail; the window listeners still follow the drag.
    }
    const move = (pointerEvent: PointerEvent) => {
      if (!moving && Math.hypot(pointerEvent.clientX - originX, pointerEvent.clientY - originY) < 4) return
      moving = true
      movedRef.current = true
      const box = planeBox()
      const left = Math.min(originX, pointerEvent.clientX) - box.left
      const top = Math.min(originY, pointerEvent.clientY) - box.top
      const width = Math.abs(pointerEvent.clientX - originX)
      const height = Math.abs(pointerEvent.clientY - originY)
      setMarquee({ left, top, width, height })
    }
    const up = (pointerEvent: PointerEvent) => {
      sheet.removeEventListener("pointermove", move)
      sheet.removeEventListener("pointerup", up)
      sheet.removeEventListener("pointercancel", up)
      setMarquee(null)
      if (!moving) return
      const box = planeBox()
      const left = Math.min(originX, pointerEvent.clientX) - box.left
      const top = Math.min(originY, pointerEvent.clientY) - box.top
      const hits = clipsInBox(projectRef.current, liveRef.current.duration, box.width, stackLayout(projectRef.current.tracks, lanesRef.current?.clientHeight ?? 0), {
        left,
        top,
        right: left + Math.abs(pointerEvent.clientX - originX),
        bottom: top + Math.abs(pointerEvent.clientY - originY),
      })
      if (!shift) {
        liveRef.current.onSelectIds(hits)
        return
      }
      const clipIds = new Set(projectRef.current.clips.map((clip) => clip.id))
      const current = liveRef.current.selectedIds.filter((id) => clipIds.has(id))
      liveRef.current.onSelectIds([...current, ...hits.filter((id) => !current.includes(id))])
    }
    sheet.addEventListener("pointermove", move)
    sheet.addEventListener("pointerup", up)
    sheet.addEventListener("pointercancel", up)
  }

  const onPlayheadPointerDown = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (projectRef.current.clips.length === 0) return
    event.stopPropagation()
    const target = event.currentTarget
    try {
      target.setPointerCapture(event.pointerId)
    } catch {
      // Pointer capture can fail; the window listeners still follow the drag.
    }
    const move = (pointerEvent: PointerEvent) => {
      nudgeScroll(pointerEvent.clientX)
      const next = readPointer(pointerEvent.clientX, pointerEvent.clientY)
      if (next) liveRef.current.onSeek(next.at)
    }
    const up = () => {
      target.removeEventListener("pointermove", move)
      target.removeEventListener("pointerup", up)
      target.removeEventListener("pointercancel", up)
    }
    target.addEventListener("pointermove", move)
    target.addEventListener("pointerup", up)
    target.addEventListener("pointercancel", up)
    const next = readPointer(event.clientX, event.clientY)
    if (next) liveRef.current.onSeek(next.at)
  }

  const view = preview ?? project
  const layout = stackLayout(view.tracks, area)
  const laneStyle = { gap: layout.gap, paddingTop: layout.padTop, paddingBottom: layout.padBottom }
  const insertTop = (insertAbove: string | null) => {
    let y = layout.stackTop
    for (const track of view.tracks) {
      if (track.id === insertAbove) return y
      y += laneHeight(track) + layout.gap
    }
    return y
  }

  const mainTop = () => {
    let y = layout.stackTop
    for (const track of view.tracks) {
      if (track.main) return y
      y += laneHeight(track) + layout.gap
    }
    return null
  }

  const clipKinds = selectedIds.flatMap((id) => {
    const clip = project.clips.find((item) => item.id === id)
    if (!clip) return []
    const kind = project.tracks.find((track) => track.id === clip.trackId)?.kind
    return kind ? [kind] : []
  })
  const allows = (kinds: TrackKind[]) => clipKinds.length > 0 && clipKinds.every((kind) => kinds.includes(kind))
  const canDelete = selectedIds.length > 0
  const pictureNote = (message: string) => onProject(project, message)
  const onDelete = () => {
    const result = deleteSelection(project, selectedIds)
    onProject(result.state, result.note ?? null)
    onSelectIds(result.selectIds)
  }

  const transitions = visibleTransitions(view)
  const menuClip = menu ? project.clips.find((clip) => clip.id === menu.id) ?? null : null
  const menuTrack = menuClip ? project.tracks.find((track) => track.id === menuClip.trackId) ?? null : null
  const ticks = rulerTicks(rulerDuration, rulerWidth, fps, project.settings.dropFrame)
  const cutTop = mainTop()

  return (
    <section ref={rootRef} className="timeline" aria-label="Timeline" style={{ height }}>
      <TimelineToast note={note} />
      <div className="tl-toolbar" role="toolbar" aria-label="Timeline tools">
        <Tip label="Undo">
          <button type="button" className="tl-tool" aria-label="Undo" disabled={!canUndo} onClick={onUndo}>
            <Icon name="undo" />
          </button>
        </Tip>
        <Tip label="Redo">
          <button type="button" className="tl-tool" aria-label="Redo" disabled={!canRedo} onClick={onRedo}>
            <Icon name="redo" />
          </button>
        </Tip>
        <span className="tl-sep" />
        <Tip label="Select">
          <button type="button" className="tl-tool" aria-label="Select" aria-pressed={tool === "select"} data-active={tool === "select" ? "true" : "false"} onClick={() => setTool("select")}>
            <Icon name="select" />
          </button>
        </Tip>
        <Tip label="Split">
          <button type="button" className="tl-tool" aria-label="Split" aria-pressed={tool === "split"} data-active={tool === "split" ? "true" : "false"} disabled={empty} onClick={() => setTool("split")}>
            <Icon name="split" />
          </button>
        </Tip>
        <Tip label="Delete">
          <button type="button" className="tl-tool" aria-label="Delete" disabled={!canDelete} onClick={onDelete}>
            <Icon name="delete" />
          </button>
        </Tip>
        <Tip label="Freeze">
          <button type="button" className="tl-tool" aria-label="Freeze" disabled={!allows(videoKinds)} onClick={() => pictureNote("The prototype does not freeze a frame.")}>
            <Icon name="freeze" />
          </button>
        </Tip>
        <Tip label="Reverse">
          <button type="button" className="tl-tool" aria-label="Reverse" disabled={!allows(timedKinds)} onClick={() => pictureNote("The prototype does not reverse a clip.")}>
            <Icon name="reverse" />
          </button>
        </Tip>
        <Tip label="Mirror">
          <button type="button" className="tl-tool" aria-label="Mirror" disabled={!allows(pictureKinds)} onClick={() => pictureNote("The prototype does not mirror a clip.")}>
            <Icon name="mirror" />
          </button>
        </Tip>
        <Tip label="Rotate">
          <button type="button" className="tl-tool" aria-label="Rotate" disabled={!allows(pictureKinds)} onClick={() => pictureNote("The prototype does not rotate a clip.")}>
            <Icon name="rotate" />
          </button>
        </Tip>
        <Tip label="Crop">
          <button type="button" className="tl-tool" aria-label="Crop" disabled={!allows(framedKinds)} onClick={() => pictureNote("The prototype does not crop a clip.")}>
            <Icon name="crop" />
          </button>
        </Tip>
        <Tip label="Transcribe">
          <button type="button" className="tl-tool" aria-label="Transcribe" disabled={!allows(timedKinds)} onClick={() => pictureNote("The prototype does not transcribe a clip.")}>
            <Icon name="transcribe" />
          </button>
        </Tip>
        <Tip label="Split scenes">
          <button type="button" className="tl-tool" aria-label="Split scenes" disabled={!allows(videoKinds)} onClick={() => pictureNote("The prototype does not split scenes.")}>
            <Icon name="splitScenes" />
          </button>
        </Tip>
        <span className="tl-sep" />
        <Tip label="Marker">
          <button type="button" className="tl-tool" aria-label="Marker" aria-pressed={tool === "marker"} data-active={tool === "marker" ? "true" : "false"} disabled={empty} onClick={() => setTool("marker")}>
            <Icon name="marker" />
          </button>
        </Tip>
        <Tip label="Beat">
          <button type="button" className="tl-tool" aria-label="Beat" aria-pressed={tool === "beat"} data-active={tool === "beat" ? "true" : "false"} disabled={empty} onClick={() => setTool("beat")}>
            <Icon name="beat" />
          </button>
        </Tip>
        <span className="tl-sep" />
        <Tip label={snap ? "Snap on" : "Snap off"}>
          <button type="button" className="tl-tool" aria-label={snap ? "Snap on" : "Snap off"} aria-pressed={snap} data-active={snap ? "true" : "false"} onClick={() => setSnap((value) => !value)}>
            <Icon name="snap" />
          </button>
        </Tip>
        <Tip label={project.magnet ? "Main track magnet on" : "Main track magnet off"}>
        <button
          type="button"
          className="tl-tool"
          aria-label={project.magnet ? "Main track magnet on" : "Main track magnet off"}
          aria-pressed={project.magnet}
          data-active={project.magnet ? "true" : "false"}
          onClick={() => {
            const main = project.tracks.find((track) => track.main && track.locked)
            if (main) {
              onProject(
                project,
                project.magnet
                  ? "Main track is locked. Couldn't turn off main track magnet."
                  : "Main track is locked. Couldn't turn on main track magnet.",
              )
              return
            }
            onProject(setMagnet(project, !project.magnet))
          }}
        >
          <Icon name="magnet" />
        </button>
        </Tip>
        <Tip label={project.linkage ? "Linkage on" : "Linkage off"}>
        <button
          type="button"
          className="tl-tool"
          aria-label={project.linkage ? "Linkage on" : "Linkage off"}
          aria-pressed={project.linkage}
          data-active={project.linkage ? "true" : "false"}
          onClick={() => onProject(setLinkage(project, !project.linkage))}
        >
          <Icon name="linkage" />
        </button>
        </Tip>
        <Tip label={project.guidesVisible ? "Hide reference lines" : "Show reference lines"}>
        <button
          type="button"
          className="tl-tool"
          aria-label={project.guidesVisible ? "Hide reference lines" : "Show reference lines"}
          aria-pressed={project.guidesVisible}
          data-active={project.guidesVisible ? "true" : "false"}
          onClick={() => onProject(setGuidesVisible(project, !project.guidesVisible))}
        >
          <Icon name="guides" />
        </button>
        </Tip>
        <Tip label={project.guidesLocked ? "Unlock reference lines" : "Lock reference lines"}>
        <button
          type="button"
          className="tl-tool"
          aria-label={project.guidesLocked ? "Unlock reference lines" : "Lock reference lines"}
          aria-pressed={project.guidesLocked}
          data-active={project.guidesLocked ? "true" : "false"}
          onClick={() => onProject(setGuidesLocked(project, !project.guidesLocked))}
        >
          <Icon name="lock" />
        </button>
        </Tip>
        <div className="tl-zoom">
          <Tip label="Zoom to fit">
          <button
            type="button"
            className="tl-tool"
            aria-label="Zoom to fit"
            disabled={empty}
            onClick={() => {
              if (empty || laneWidth <= 0 || duration <= 0) return
              tuned.current = true
              zoomAnchor.current = 0
              const fit = clampPixels((laneWidth * FIT_WIDTH_RATIO) / duration, laneWidth, duration, ceiling)
              pixelsRef.current = fit
              setPixels(fit)
            }}
          >
            <Icon name="zoomFit" />
          </button>
          </Tip>
          <Tip label="Zoom out">
          <button
            type="button"
            className="tl-tool"
            aria-label="Timeline zoom out"
            disabled={empty || pixels <= zoomFloor(laneWidth, duration, ceiling) + 0.01}
            onClick={() => applyZoom(pixels * 0.5)}
          >
            <Icon name="zoomOut" />
          </button>
          </Tip>
          <Tip label="Zoom in">
          <button
            type="button"
            className="tl-tool"
            aria-label="Timeline zoom in"
            disabled={empty || pixels >= ceiling - 0.01}
            onClick={() => applyZoom(pixels * 2)}
          >
            <Icon name="zoomIn" />
          </button>
          </Tip>
        </div>
      </div>
      <div className="tl-ruler">
        <div className="tl-ruler-head" onPointerDown={(event) => pullGuide(event, "row")} />
        <div className="tl-ruler-cover" />
        <div className="tl-ruler-view">
          <div className="ruler" ref={rulerRef} style={rulerWidth > 0 ? { width: rulerWidth } : undefined} onPointerDown={(event) => pullGuide(event, "time")}>
            {ticks.map((tick) => (
              <span
                key={tick.frame}
                className="tick"
                data-major={tick.major ? "true" : "false"}
                style={{ left: `${(tick.time / rulerDuration) * 100}%` }}
              >
                {tick.label ? <span className="tick-label">{tick.label}</span> : null}
              </span>
            ))}
            {view.markers
              .filter((marker) => !marker.clipId)
              .map((marker) => (
                <span
                  key={marker.id}
                  className="marker"
                  data-tip={`${marker.beat ? "Beat" : "Marker"} · ${formatTime(marker.time)}`}
                  style={{ left: `${(marker.time / rulerDuration) * 100}%`, background: marker.color }}
                  onPointerDown={(event) => {
                    event.stopPropagation()
                    onProject(removeMarker(projectRef.current, marker.id))
                  }}
                />
              ))}
          </div>
        </div>
      </div>
      <div className="tl-tracks">
        <div
          className="tl-head"
          ref={headRef}
          onClick={() => {
            if (liveRef.current.tool === "select") liveRef.current.onSelectIds([])
          }}
        >
          <div className="lane-stack" style={laneStyle}>
          {view.tracks.map((track) => (
            <div key={track.id} className="track-label" data-main={track.main ? "true" : "false"} data-locked={track.locked ? "true" : "false"}>
              <Icon name={trackIcon[track.kind]} size={14} />
              <span>{track.main ? "Main" : trackKindLabel[track.kind]}</span>
              <button
                type="button"
                className="track-lock"
                aria-label={track.locked ? "Unlock track" : "Lock track"}
                data-locked={track.locked ? "true" : "false"}
                onClick={(event) => {
                  event.stopPropagation()
                  const locked = !track.locked
                  onProject(setTrackLocked(project, track.id, locked), locked ? "Track locked" : "Track unlocked")
                }}
              >
                <Icon name="lock" size={12} />
              </button>
            </div>
          ))}
          </div>
        </div>
        <div className="tl-cover" />
        <div
          ref={lanesRef}
          className="tl-lanes"
          onScroll={syncScroll}
          onPointerDown={onSheetPointerDown}
          onClick={(event) => {
            if (movedRef.current) {
              movedRef.current = false
              return
            }
            if (liveRef.current.tool === "select") liveRef.current.onSelectIds([])
            const rect = sheetRef.current?.getBoundingClientRect()
            if (!rect || rect.width <= 0) return
            const at = Math.min(duration, Math.max(0, ((event.clientX - rect.left) / rect.width) * duration))
            if (tool === "marker") {
              onProject(addMarker(project, at, null, false).state)
              return
            }
            if (tool === "beat") {
              onProject(project, "Beat marks sit on audio.")
              return
            }
            if (empty) return
            onSeek(at)
          }}
          onDragOver={onDragOver}
          onDrop={onDrop}
          onDragLeave={(event) => {
            if (event.currentTarget.contains(event.relatedTarget as Node | null)) return
            setCutAt(null)
            setAdsorb(null)
            setHover((current) => (current?.movingId ? current : null))
          }}
        >
          <div className="lane-stack lane-backs" style={laneStyle}>
          {view.tracks.map((track) => {
            const landing = hover?.plan.ok && !hover.plan.create && hover.plan.trackId === track.id
            const refused = hover?.plan.ok === false && hover.plan.hoverTrackId === track.id
            const empty = track.main && !view.clips.some((clip) => clip.trackId === track.id)
            return (
              <div
                key={track.id}
                className="lane"
                data-main={track.main ? "true" : "false"}
                data-empty={empty ? "true" : "false"}
                data-drop={landing ? "yes" : refused ? "no" : "none"}
              >
                {empty && (
                  <div className="lane-empty">
                    <Icon name="video" size={14} />
                    <span>Drag a clip here</span>
                  </div>
                )}
              </div>
            )
          })}
          </div>
          <div ref={sheetRef} className="tl-plane" style={contentWidth > 0 ? { width: contentWidth } : undefined}>
          <div className="lane-stack" style={laneStyle}>
          {view.tracks.map((track) => (
              <div key={track.id} className="lane" data-main={track.main ? "true" : "false"}>
                {view.clips
                  .filter((clip) => clip.trackId === track.id)
                  .map((clip) => {
                    const start = clip.start
                    const away = drag?.id === clip.id
                    return (
                      <button
                        key={clip.id}
                        type="button"
                        className="clip"
                        data-tip={`${clip.name} · ${formatTime(clip.start)}–${formatTime(clip.start + clip.duration)}`}
                        data-kind={track.kind}
                        data-main={track.main ? "true" : "false"}
                        data-tone={clip.tone}
                        data-selected={selectedIds.includes(clip.id) ? "true" : "false"}
                        data-away={away ? "true" : "false"}
                        style={{
                          left: `calc(${(start / duration) * 100}% + 1px)`,
                          width: `calc(${(clip.duration / duration) * 100}% - 2px)`,
                        }}
                        onPointerDown={(event) => onClipPointerDown(event, clip)}
                        onContextMenu={(event) => onClipMenu(event, clip)}
                        onClick={(event) => event.stopPropagation()}
                      >
                        <span>{clip.name}</span>
                        {project.silencesRemoved && clip.tone === "picture" && (
                          <>
                            <i className="notch" style={{ left: "22%" }} />
                            <i className="notch" style={{ left: "48%" }} />
                            <i className="notch" style={{ left: "71%" }} />
                          </>
                        )}
                        <span className="clip-edge" data-edge="in" onPointerDown={(event) => onTrimPointerDown(event, clip, "in")} />
                        <span className="clip-edge" data-edge="out" onPointerDown={(event) => onTrimPointerDown(event, clip, "out")} />
                        {view.markers
                          .filter((marker) => marker.clipId === clip.id)
                          .map((marker) => (
                            <span
                              key={marker.id}
                              className={marker.beat ? "beat" : "marker"}
                              data-tip={`${marker.beat ? "Beat" : "Marker"} · ${formatTime(marker.time)}`}
                              style={{
                                left: `${((marker.time - clip.start) / Math.max(0.1, clip.duration)) * 100}%`,
                                background: marker.color,
                              }}
                              onPointerDown={(event) => {
                                event.stopPropagation()
                                onProject(removeMarker(projectRef.current, marker.id))
                              }}
                            />
                          ))}
                      </button>
                    )
                  })}
                {track.main &&
                  transitions.map((transition) => (
                    <button
                      key={transition.id}
                      type="button"
                      className="transition"
                      data-tip={`${transition.name} · ${formatTime(transition.duration)}`}
                      data-selected={selectedIds.includes(transition.id) ? "true" : "false"}
                      style={{ left: `${(transition.at / duration) * 100}%`, width: `max(22px, ${(transition.duration / duration) * 100}%)` }}
                      aria-label={transition.name}
                      onPointerDown={(event) => {
                        event.stopPropagation()
                        if (liveRef.current.tool === "split") {
                          liveRef.current.onProject(projectRef.current, "Transitions cannot be split.")
                          return
                        }
                        liveRef.current.onSelect(transition.id)
                      }}
                      onClick={(event) => event.stopPropagation()}
                    >
                      {transition.name}
                    </button>
                  ))}
              </div>
          ))}
          </div>
          {hover?.plan.ok && hover.plan.create && <div className="insert-line" style={{ top: insertTop(hover.plan.insertAbove) }} />}
          {cutAt !== null && cutTop !== null && <div className="cut-line" style={{ top: cutTop, left: `${(cutAt / duration) * 100}%` }} />}
          {project.guidesVisible &&
            view.guides.map((guide) => {
              const at = guideDrag?.id === guide.id ? guideDrag.at : guide.at
              return (
                <div
                  key={guide.id}
                  className="guide-line"
                  data-tip={guide.axis === "time" ? `Reference line · ${formatTime(guide.at)}` : "Reference line"}
                  data-axis={guide.axis}
                  data-selected={selectedIds.includes(guide.id) ? "true" : "false"}
                  style={guide.axis === "time" ? { left: `${(at / duration) * 100}%` } : { top: layout.stackTop + at * layout.stackHeight }}
                  onPointerDown={(event) => onGuidePointerDown(event, guide)}
                  onClick={(event) => event.stopPropagation()}
                />
              )
            })}
          {draftGuide && (
            <div
              className="guide-line"
              data-axis={draftGuide.axis}
              data-draft="true"
              style={draftGuide.axis === "time" ? { left: `${(draftGuide.at / duration) * 100}%` } : { top: layout.stackTop + draftGuide.at * layout.stackHeight }}
            />
          )}
          {adsorb !== null && <div className="adsorb-line" style={{ left: `${(adsorb / duration) * 100}%` }} />}
          {marquee && <div className="marquee" style={marquee} />}
          {drag && (
            <div
              className="clip"
              data-drag="true"
              data-kind={drag.kind}
              data-main={drag.main ? "true" : "false"}
              data-tone={drag.tone}
              data-selected="true"
              style={{
                left: `calc(${(drag.start / duration) * 100}% + 1px)`,
                width: `calc(${(drag.duration / duration) * 100}% - 2px)`,
                top: drag.top,
              }}
            >
              <span>{drag.name}</span>
            </div>
          )}
        </div>
      </div>
      </div>
      <div className="playhead-frame" ref={playheadFrameRef}>
        <div ref={playheadRef} className="playhead" data-off={empty ? "true" : "false"}>
          <Tip label="Playhead">
            <button type="button" className="playhead-cap" aria-label="Playhead" disabled={empty} onPointerDown={onPlayheadPointerDown} />
          </Tip>
        </div>
      </div>
      {menuClip && menu && (
        <ClipMenu
          x={menu.x}
          y={menu.y}
          items={menuFor(menuTrack?.kind ?? "video")}
          onClose={() => setMenu(null)}
          onPick={onMenuPick}
        />
      )}
    </section>
  )
}

function clipsInBox(
  state: ProjectState,
  duration: number,
  sheetWidth: number,
  layout: StackLayout,
  box: { left: number; top: number; right: number; bottom: number },
) {
  const rank = new Map(state.tracks.map((track, index) => [track.id, index]))
  const hits = state.clips.filter((clip) => {
    const bounds = clipBounds(state.tracks, clip, duration, sheetWidth, layout)
    if (!bounds) return false
    return bounds.right >= box.left && bounds.left <= box.right && bounds.bottom >= box.top && bounds.top <= box.bottom
  })
  hits.sort((a, b) => (rank.get(a.trackId) ?? 0) - (rank.get(b.trackId) ?? 0) || a.start - b.start)
  const primary = hits[0]?.id
  if (!primary) return []
  return [...hits.map((clip) => clip.id).filter((id) => id !== primary), primary]
}

function clipBounds(tracks: Track[], clip: Clip, duration: number, sheetWidth: number, layout: StackLayout) {
  let y = layout.stackTop
  for (const track of tracks) {
    const lane = laneHeight(track)
    if (track.id === clip.trackId) {
      const top = y + (track.main ? 8 : 6)
      const height = track.main ? 48 : 28
      const left = (clip.start / duration) * sheetWidth
      const width = (clip.duration / duration) * sheetWidth
      return { left, top, right: left + width, bottom: top + height }
    }
    y += lane + layout.gap
  }
  return null
}

const LANE_GAP = 8

type StackLayout = {
  stackTop: number
  stackHeight: number
  padTop: number
  padBottom: number
  gap: number
  scrollTop: number
}

function laneHeight(track: { main?: boolean }) {
  return track.main ? 64 : 40
}

function stackLayout(tracks: Track[], areaHeight: number): StackLayout {
  const mainIndex = tracks.findIndex((track) => track.main)
  let above = 0
  let below = 0
  let mainH = 64
  tracks.forEach((track, index) => {
    const height = laneHeight(track)
    if (track.main) {
      mainH = height
      return
    }
    if (mainIndex >= 0 && index > mainIndex) below += height
    else above += height
  })
  const gapsBefore = mainIndex > 0 ? mainIndex : 0
  const afterCount = mainIndex >= 0 ? Math.max(0, tracks.length - mainIndex - 1) : 0
  const aboveBlock = above + gapsBefore * LANE_GAP
  const belowBlock = below + afterCount * LANE_GAP
  const padTop = Math.max(0, areaHeight / 2 - mainH / 2 - aboveBlock)
  const padBottom = Math.max(0, areaHeight / 2 - mainH / 2 - belowBlock)
  const stackHeight = above + mainH + below + Math.max(0, tracks.length - 1) * LANE_GAP
  return {
    stackTop: padTop,
    stackHeight,
    padTop,
    padBottom,
    gap: LANE_GAP,
    scrollTop: Math.max(0, aboveBlock + mainH / 2 - areaHeight / 2),
  }
}

function visibleTransitions(state: ProjectState) {
  const main = state.tracks.find((track) => track.main)
  if (!main) return []
  return state.transitions.flatMap((transition) => {
    const left = state.clips.find((clip) => clip.id === transition.leftId && clip.trackId === main.id)
    if (!left) return []
    const cut = left.start + left.duration
    const right = state.clips.find((clip) => clip.trackId === main.id && clip.id !== left.id && Math.abs(clip.start - cut) < 0.08)
    if (!right) return []
    return [{ ...transition, at: cut }]
  })
}
