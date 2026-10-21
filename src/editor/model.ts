export const DURATION = 10

export type ToolId =
  | "media"
  | "audio"
  | "text"
  | "stickers"
  | "effects"
  | "transitions"
  | "filters"
  | "adjustments"
  | "templates"

export type Tone = "picture" | "title" | "sound" | "generated" | "sticker" | "effect" | "filter" | "adjust"

export const trackKindLabel = {
  video: "Video",
  audio: "Audio",
  sticker: "Sticker",
  text: "Text",
  effect: "Effect",
  filter: "Filter",
  adjust: "Adjust",
  text_to_video_text: "Text to video",
  text_to_video_text_audio: "Text to video audio",
  composition: "Composition",
  plugin_effect: "Plugin effect",
  adcube: "Ad",
  mixed: "Mixed",
} as const

export type TrackKind = keyof typeof trackKindLabel

const layerOrder = ["plugin", "graphic", "effect", "filter", "adjust", "overlay", "main", "audio", "adcube"] as const

function layerOf(track: { kind: TrackKind; main?: boolean }) {
  if (track.main) return "main"
  if (track.kind === "plugin_effect") return "plugin"
  if (track.kind === "text" || track.kind === "sticker" || track.kind === "text_to_video_text") return "graphic"
  if (track.kind === "effect") return "effect"
  if (track.kind === "filter") return "filter"
  if (track.kind === "adjust") return "adjust"
  if (track.kind === "video" || track.kind === "mixed" || track.kind === "composition") return "overlay"
  if (track.kind === "audio" || track.kind === "text_to_video_text_audio") return "audio"
  return "adcube"
}

export type Track = {
  id: string
  kind: TrackKind
  main?: boolean
  locked?: boolean
}

export type Clip = {
  id: string
  trackId: string
  name: string
  start: number
  duration: number
  source: number
  trimIn: number
  tone: Tone
  scene?: string
  assetId?: string
}

export type Transition = {
  id: string
  name: string
  leftId: string
  duration: number
}

export type Marker = {
  id: string
  time: number
  color: string
  beat: boolean
  clipId: string | null
}

export type Guide = {
  id: string
  axis: "time" | "row"
  at: number
}

export const frameRates = ["23.976", "24", "25", "29.97", "30", "50", "59.94", "60"] as const
export type FrameRate = (typeof frameRates)[number]
export const colorSpaces = ["Rec. 709 SDR", "Rec. 2020 HLG", "Rec. 2100 PQ"] as const
export type ColorSpace = (typeof colorSpaces)[number]
export const aspectRatios = ["Original", "16:9", "9:16", "1:1", "4:5", "4:3"] as const
export type AspectRatio = (typeof aspectRatios)[number]
export const projectResolutions = ["Adapted", "720p", "1080p", "2K", "4K"] as const
export type ProjectResolution = (typeof projectResolutions)[number]

export type ProjectSettings = {
  name: string
  importMaterial: "copy" | "keep"
  colorSpace: ColorSpace
  freeLayer: boolean
  proxy: boolean
  ratio: AspectRatio
  resolution: ProjectResolution
  frameRate: FrameRate
  /** SMPTE drop-frame labels. Only 29.97 and 59.94 change the count. */
  dropFrame: boolean
}

const frameRateLabels: Record<FrameRate, string> = {
  "23.976": "23.98 fps",
  "24": "24.00 fps",
  "25": "25.00 fps",
  "29.97": "29.97 fps",
  "30": "30.00 fps",
  "50": "50.00 fps",
  "59.94": "59.94 fps",
  "60": "60.00 fps",
}

export function frameRateLabel(rate: FrameRate) {
  return frameRateLabels[rate]
}

export function frameRateFps(rate: FrameRate) {
  if (rate === "23.976") return 23.976
  if (rate === "29.97") return 29.97
  if (rate === "59.94") return 59.94
  return Number(rate)
}

export function projectPath(name: string) {
  const trimmed = name.trim() || "Untitled"
  return `Projects/${trimmed}`
}

export const initialSettings: ProjectSettings = {
  name: "Amina & Jules",
  importMaterial: "keep",
  colorSpace: "Rec. 709 SDR",
  freeLayer: false,
  proxy: false,
  ratio: "Original",
  resolution: "Adapted",
  frameRate: "30",
  dropFrame: false,
}

/** 29.97 drops 2 frame numbers a minute. 59.94 drops 4. Every tenth minute drops none. */
export function dropFrameCount(rate: FrameRate) {
  if (rate === "29.97") return 2
  if (rate === "59.94") return 4
  return 0
}

export function supportsDropFrame(rate: FrameRate) {
  return dropFrameCount(rate) > 0
}

export type ProjectState = {
  tracks: Track[]
  clips: Clip[]
  silencesRemoved: boolean
  magnet: boolean
  linkage: boolean
  transitions: Transition[]
  markers: Marker[]
  guides: Guide[]
  guidesVisible: boolean
  guidesLocked: boolean
  settings: ProjectSettings
}

export type Asset = {
  id: string
  tool: ToolId
  name: string
  meta: string
  swatch: string
}

export type EditorApi = {
  time: number
  duration: number
  playing: boolean
  project: ProjectState
  selectedId: string | null
  selectedIds: string[]
  tool: ToolId
  note: string | null
  onTogglePlay: () => void
  onSeek: (time: number) => void
  onStill: () => void
  onSelect: (id: string) => void
  onSelectIds: (ids: string[]) => void
  onTool: (tool: ToolId) => void
  onCommand: (command: string) => void
  onProject: (project: ProjectState, note?: string | null, selectId?: string) => void
  canUndo: boolean
  canRedo: boolean
  onUndo: () => void
  onRedo: () => void
}

export type CommandResult = {
  state: ProjectState
  note: string
  selectId?: string
  seek?: number
}

export const tools: { id: ToolId; label: string }[] = [
  { id: "media", label: "Media" },
  { id: "audio", label: "Audio" },
  { id: "text", label: "Text" },
  { id: "stickers", label: "Stickers" },
  { id: "effects", label: "Effects" },
  { id: "transitions", label: "Transitions" },
  { id: "filters", label: "Filters" },
  { id: "adjustments", label: "Adjustments" },
  { id: "templates", label: "Templates" },
]

export const suggestions = [
  { label: "Remove silences", command: "Remove all silences" },
  { label: "Insert shot at 0:32", command: "Generate a shot and insert it at 0:32" },
  { label: "Wedding film", command: "Using all these footages, make a wedding film" },
]

/**
 * Sequence rate. 30 fps is CapCut frame-rate id 5.
 * Pixels per second are CapCut's pixels-per-microsecond size times 1,000,000.
 */
export const SEQUENCE_FPS = 30

type RateId = "23.976" | "24" | "25" | "29.97" | "30" | "50" | "59.94" | "60"

const rates: { id: RateId; fps: number; nominal: number; clockMinors: number }[] = [
  { id: "23.976", fps: 23.976, nominal: 24, clockMinors: 12 },
  { id: "24", fps: 24, nominal: 24, clockMinors: 12 },
  { id: "25", fps: 25, nominal: 25, clockMinors: 5 },
  { id: "29.97", fps: 29.97, nominal: 30, clockMinors: 10 },
  { id: "30", fps: 30, nominal: 30, clockMinors: 10 },
  { id: "50", fps: 50, nominal: 50, clockMinors: 10 },
  { id: "59.94", fps: 59.94, nominal: 60, clockMinors: 10 },
  { id: "60", fps: 60, nominal: 60, clockMinors: 10 },
]

/** Clock steps, coarse to fine. Thresholds are pixels per microsecond. */
const clockSteps: [seconds: number, threshold: number][] = [
  [108000, 2.6e-9],
  [36000, 8e-9],
  [18000, 1.6e-8],
  [10800, 2.6e-8],
  [7200, 4e-8],
  [3600, 8e-8],
  [1800, 1.6e-7],
  [600, 5e-7],
  [300, 1e-6],
  [180, 1.6e-6],
  [120, 2.5e-6],
  [60, 5e-6],
  [30, 1e-5],
  [10, 3e-5],
  [5, 6e-5],
  [3, 1e-4],
  [2, 1.5e-4],
  [1, 3e-4],
]

const frameMinors: Record<number, number> = {
  30: 10,
  25: 5,
  20: 10,
  15: 5,
  12: 6,
  10: 10,
  8: 4,
  6: 6,
  5: 5,
  4: 4,
  3: 3,
  2: 2,
}

/** Frame steps installed for each rate, coarse to fine. Thresholds are pixels per microsecond. */
const frameSteps: Record<RateId, [frames: number, threshold: number][]> = {
  "23.976": [
    [12, 0.0006],
    [8, 0.0008],
    [6, 0.0009],
    [4, 0.001],
    [3, 0.002],
    [2, 0.0048],
  ],
  "24": [
    [12, 0.0006],
    [8, 0.0008],
    [4, 0.001],
    [3, 0.002],
    [2, 0.0048],
  ],
  "25": [[5, 0.005]],
  "29.97": [
    [15, 0.0006],
    [10, 0.0008],
    [6, 0.0009],
    [5, 0.001],
    [3, 0.0013],
    [2, 0.006],
  ],
  "30": [
    [15, 0.0006],
    [10, 0.0008],
    [5, 0.001],
    [3, 0.0013],
    [2, 0.006],
  ],
  "50": [
    [25, 0.0006],
    [10, 0.0009],
    [5, 0.005],
    [2, 0.01],
  ],
  "59.94": [
    [30, 0.0006],
    [20, 0.0008],
    [15, 0.00085],
    [12, 0.0009],
    [10, 0.001],
    [6, 0.0013],
    [5, 0.003],
    [4, 0.006],
    [3, 0.009],
    [2, 0.012],
  ],
  "60": [
    [30, 0.0006],
    [20, 0.0008],
    [15, 0.00085],
    [10, 0.001],
    [5, 0.003],
    [3, 0.009],
    [2, 0.012],
  ],
}

type RulerStep = { frames: number; interval: number; threshold: number; frame: boolean }

function rateFor(fps: number) {
  return rates.reduce((best, rate) => (Math.abs(rate.fps - fps) < Math.abs(best.fps - fps) ? rate : best))
}

function stepsFor(fps: number) {
  const rate = rateFor(fps)
  const steps: RulerStep[] = clockSteps.map(([seconds, threshold]) => ({
    frames: seconds * rate.nominal,
    interval: rate.clockMinors,
    threshold,
    frame: false,
  }))
  for (const [frames, threshold] of frameSteps[rate.id]) {
    steps.push({ frames, interval: frameMinors[frames], threshold, frame: true })
  }
  return { nominal: rate.nominal, steps }
}

function chooseStep(pixelsPerSecond: number, fps: number) {
  const { nominal, steps } = stepsFor(fps)
  const pixelSize = pixelsPerSecond / 1_000_000
  const step = steps.find((item) => item.threshold >= pixelSize) ?? steps[steps.length - 1]
  return { nominal, step }
}

/** Largest zoom: the finest step's threshold, in pixels per second. At 30 fps that is the two-frame step. */
export function zoomCeiling(fps = SEQUENCE_FPS) {
  const { steps } = stepsFor(fps)
  return steps[steps.length - 1].threshold * 1_000_000
}

function pad(value: number) {
  return value.toString().padStart(2, "0")
}

function clockLabel(frame: number, nominal: number) {
  const whole = Math.max(0, Math.round(frame / nominal))
  const s = whole % 60
  const m = Math.floor(whole / 60) % 60
  const h = Math.floor(whole / 3600)
  if (h >= 1) return `${pad(h)}:${pad(m)}:${pad(s)}`
  return `${pad(m)}:${pad(s)}`
}

function tickLabel(frame: number, nominal: number, step: RulerStep) {
  if (!step.frame || frame % nominal === 0) return clockLabel(frame, nominal)
  const withinSecond = frame % nominal
  return `${withinSecond}f`
}

/** SMPTE drop-frame `HH:MM:SS;FF` from a nominal frame index. */
function dropFrameLabel(frame: number, nominal: number, drop: number) {
  const perMinute = nominal * 60 - drop
  const perTen = nominal * 60 * 10 - drop * 9
  const tens = Math.floor(frame / perTen)
  const within = frame % perTen
  const adjusted = within > drop ? frame + drop * 9 * tens + drop * Math.floor((within - drop) / perMinute) : frame + drop * 9 * tens
  const f = adjusted % nominal
  const whole = Math.floor(adjusted / nominal)
  const s = whole % 60
  const m = Math.floor(whole / 60) % 60
  const h = Math.floor(whole / 3600)
  return `${pad(h)}:${pad(m)}:${pad(s)};${pad(f)}`
}

/** Seconds snapped to the frame the ruler ticks use. */
export function frameSeconds(seconds: number, fps: number) {
  const nominal = rateFor(fps).nominal
  const frame = Math.max(0, Math.floor(seconds * nominal + 1e-6))
  return frame / nominal
}

/** Playhead and duration, `HH:MM:SS:FF`. Drop frame uses a semicolon. */
export function formatTimecode(seconds: number, rate: FrameRate, dropFrame = false) {
  const nominal = rateFor(frameRateFps(rate)).nominal
  const frame = Math.max(0, Math.floor(seconds * nominal + 1e-6))
  const drop = dropFrame ? dropFrameCount(rate) : 0
  if (drop > 0) return dropFrameLabel(frame, nominal, drop)
  const f = frame % nominal
  const whole = Math.floor(frame / nominal)
  const s = whole % 60
  const m = Math.floor(whole / 60) % 60
  const h = Math.floor(whole / 3600)
  return `${pad(h)}:${pad(m)}:${pad(s)}:${pad(f)}`
}

export type RulerTick = { frame: number; time: number; major: boolean; label: string | null }

export function rulerTicks(duration: number, width: number, fps = SEQUENCE_FPS, dropFrame = false): RulerTick[] {
  if (!(width > 0) || !(duration > 0)) return []
  const rate = rateFor(fps)
  const drop = dropFrame ? dropFrameCount(rate.id) : 0
  const { nominal, step } = chooseStep(width / duration, fps)
  const minor = step.frames / step.interval
  const last = Math.round(Math.max(duration, 1 / nominal) * nominal)
  const ticks: RulerTick[] = []
  for (let frame = 0; frame <= last; frame += minor) {
    const major = frame % step.frames === 0
    ticks.push({
      frame,
      time: frame / nominal,
      major,
      label: major ? (drop > 0 ? dropFrameLabel(frame, nominal, drop) : tickLabel(frame, nominal, step)) : null,
    })
  }
  return ticks
}

export function timelineDuration(clips: Clip[]) {
  const end = clips.reduce((max, clip) => Math.max(max, clip.start + clip.duration), 0)
  return Math.max(DURATION, end)
}

export const scenes: Record<string, [string, string, string]> = {
  ceremony: ["#2c1b12", "#e09a62", "#100e0c"],
  vows: ["#261820", "#c46a72", "#110c10"],
  portraits: ["#172430", "#8aa4b8", "#0c1014"],
  dance: ["#2a2012", "#e2a45c", "#12100c"],
  "stock-aisle": ["#1c2418", "#8eae72", "#10140e"],
  "stock-dusk": ["#1a1428", "#6a5a98", "#0c0a12"],
  "stock-shore": ["#102028", "#6aa4b8", "#0c1214"],
  "stock-hands": ["#2a2018", "#d7b48a", "#14100c"],
  "stock-sparks": ["#24180e", "#e08a3a", "#120e0a"],
  "stock-hall": ["#1a1c22", "#8a909c", "#101216"],
  rings: ["#2a241c", "#d7c4a2", "#14120e"],
  exit: ["#2a1c14", "#c9835a", "#140e0c"],
}

export const stockFootage: { id: string; name: string; duration: number; swatch: string }[] = [
  { id: "stock-aisle", name: "Garden aisle", duration: 12, swatch: "#8eae72" },
  { id: "stock-dusk", name: "City dusk", duration: 8, swatch: "#6a5a98" },
  { id: "stock-shore", name: "Open shore", duration: 15, swatch: "#6aa4b8" },
  { id: "stock-hands", name: "Two hands", duration: 6, swatch: "#d7b48a" },
  { id: "stock-sparks", name: "Sparklers", duration: 10, swatch: "#e08a3a" },
  { id: "stock-hall", name: "Empty hall", duration: 18, swatch: "#8a909c" },
]

function bare(clip: Omit<Clip, "source" | "trimIn">, trimIn = 0, tail = 0): Clip {
  return { ...clip, trimIn, source: clip.duration + trimIn + tail }
}

export const initialProject: ProjectState = {
  silencesRemoved: false,
  magnet: true,
  linkage: true,
  transitions: [],
  markers: [],
  guides: [],
  guidesVisible: true,
  guidesLocked: false,
  settings: initialSettings,
  tracks: [{ id: "v1", kind: "video", main: true }],
  clips: [],
}

export const assets: Asset[] = [
  { id: "ceremony-card", tool: "media", name: "Ceremony", meta: "0:18", swatch: "#e09a62" },
  { id: "vows-card", tool: "media", name: "Vows", meta: "0:30", swatch: "#c46a72" },
  { id: "portraits-card", tool: "media", name: "Portraits", meta: "0:20", swatch: "#8aa4b8" },
  { id: "dance-card", tool: "media", name: "Dance", meta: "0:24", swatch: "#e2a45c" },
  { id: "rings-card", tool: "media", name: "Rings", meta: "0:08", swatch: "#d7c4a2" },
  { id: "exit-card", tool: "media", name: "Exit", meta: "0:12", swatch: "#c9835a" },
  { id: "paper-boat", tool: "audio", name: "Paper boat", meta: "1:32 · music", swatch: "#7dba9a" },
  { id: "vows-mic-card", tool: "audio", name: "Vows mic", meta: "0:26 · talk", swatch: "#5e9e86" },
  { id: "room", tool: "audio", name: "Room tone", meta: "0:40", swatch: "#3f6f62" },
  { id: "applause", tool: "audio", name: "Applause", meta: "0:06", swatch: "#9ed0b4" },
  { id: "serif", tool: "text", name: "Serif title", meta: "Rise", swatch: "#e7d7b4" },
  { id: "lower-style", tool: "text", name: "Lower third", meta: "Slide", swatch: "#efc56a" },
  { id: "end-style", tool: "text", name: "End card", meta: "Fade", swatch: "#f0e2c4" },
  { id: "kinetic", tool: "text", name: "Kinetic line", meta: "Type on", swatch: "#e4ff57" },
  { id: "ring", tool: "stickers", name: "Ring", meta: "Gold", swatch: "#e2b657" },
  { id: "flower", tool: "stickers", name: "Flower", meta: "Pressed", swatch: "#d98a93" },
  { id: "frame", tool: "stickers", name: "Frame", meta: "Hairline", swatch: "#f3f0e8" },
  { id: "confetti", tool: "stickers", name: "Confetti", meta: "Burst", swatch: "#e4ff57" },
  { id: "warm", tool: "effects", name: "Warm film", meta: "Look", swatch: "#e09a62" },
  { id: "grain", tool: "effects", name: "Fade grain", meta: "Texture", swatch: "#b7aa9a" },
  { id: "bloom", tool: "effects", name: "Bloom", meta: "Light", swatch: "#f4f0c8" },
  { id: "dip", tool: "transitions", name: "Dip to black", meta: "0.4s", swatch: "#2a2c30" },
  { id: "whip", tool: "transitions", name: "Whip", meta: "0.2s", swatch: "#d7d2c6" },
  { id: "dissolve", tool: "transitions", name: "Dissolve", meta: "0.6s", swatch: "#8d8a84" },
  { id: "leak", tool: "transitions", name: "Light leak", meta: "0.5s", swatch: "#e7a15a" },
  { id: "soft", tool: "filters", name: "Soft light", meta: "Look", swatch: "#f0d8b0" },
  { id: "film", tool: "filters", name: "Film", meta: "Look", swatch: "#c4a882" },
  { id: "cool", tool: "filters", name: "Cool", meta: "Look", swatch: "#8aa4b8" },
  { id: "exposure-card", tool: "adjustments", name: "Exposure", meta: "Whole film", swatch: "#d7d0c4" },
  { id: "contrast", tool: "adjustments", name: "Contrast", meta: "Curve", swatch: "#b7b2a8" },
  { id: "temperature", tool: "adjustments", name: "Temperature", meta: "Warm", swatch: "#e09a62" },
  { id: "wedding", tool: "templates", name: "Wedding film", meta: "Uses the bin", swatch: "#e4ff57" },
  { id: "vows-template", tool: "templates", name: "Vows", meta: "Speech", swatch: "#c46a72" },
  { id: "reception", tool: "templates", name: "Reception", meta: "Dance", swatch: "#e2a45c" },
  { id: "teaser", tool: "templates", name: "Teaser", meta: "0:15", swatch: "#8aa4b8" },
]

export function formatTime(seconds: number) {
  const safe = Math.max(0, seconds)
  const minutes = Math.floor(safe / 60)
  const remainder = safe - minutes * 60
  const whole = Math.floor(remainder)
  const tenth = Math.floor((remainder - whole) * 10)
  return `${minutes}:${whole.toString().padStart(2, "0")}.${tenth}`
}

export function findClip(clips: Clip[], id: string | null) {
  if (!id) return null
  return clips.find((clip) => clip.id === id) ?? null
}

export function clipAt(clips: Clip[], time: number, tone: Tone) {
  const hits = clips.filter(
    (clip) => clip.tone === tone && time >= clip.start && time < clip.start + clip.duration,
  )
  return hits.sort((a, b) => b.start - a.start)[0] ?? null
}

const captionLines = ["We are gathered", "for the vows", "and then we dance"]

export function addCaptions(state: ProjectState, clipId: string): { state: ProjectState; note: string; selectId?: string } {
  const clip = state.clips.find((item) => item.id === clipId)
  const from = state.tracks.find((track) => track.id === clip?.trackId)
  if (!clip || !from || (from.kind !== "video" && from.kind !== "audio" && from.kind !== "text_to_video_text_audio")) {
    return { state, note: "Transcribe a video or an audio clip." }
  }
  const mark = `caption:${clip.id}`
  const already = state.clips.find((item) => item.assetId === mark)
  if (already) return { state, note: "Captions are already on the timeline.", selectId: already.id }
  const slice = Math.max(0.4, clip.duration / captionLines.length)
  const hits = (trackId: string, start: number, duration: number, clips: Clip[]) =>
    clips.some((item) => item.trackId === trackId && start < item.start + item.duration - 0.01 && start + duration > item.start + 0.01)
  let tracks = state.tracks
  let clips = state.clips
  let track = tracks.find(
    (item) => item.kind === "text" && captionLines.every((_, index) => !hits(item.id, clip.start + index * slice, slice, clips)),
  )
  if (!track) {
    track = { id: nextTrackId(tracks), kind: "text" }
    tracks = placeTrack(tracks, track)
  }
  let selectId = ""
  captionLines.forEach((line, index) => {
    const start = clip.start + index * slice
    const duration = Math.max(0.4, Math.min(slice, clip.start + clip.duration - start))
    const id = nextClipId(clips)
    if (!selectId) selectId = id
    clips = [
      ...clips,
      bare({ id, trackId: track.id, name: line, start, duration, tone: "title", assetId: mark }),
    ]
  })
  return { state: { ...state, tracks, clips }, note: "Captions are on a text track.", selectId }
}

function bandInsertAbove(tracks: Track[], kind: TrackKind, hoverId: string | null, main = false) {
  const incoming = layerOf({ kind, main })
  const hover = hoverId ? tracks.find((track) => track.id === hoverId) : undefined
  if (hover && layerOf(hover) === incoming && incoming !== "main") return hover.id
  const want = layerOrder.indexOf(incoming)
  return tracks.find((track) => layerOrder.indexOf(layerOf(track)) >= want)?.id ?? null
}

export function placeTrack(tracks: Track[], track: Track) {
  if (tracks.some((item) => item.id === track.id)) return tracks
  const above = bandInsertAbove(tracks, track.kind, null, Boolean(track.main))
  if (!above) return [...tracks, track]
  const index = tracks.findIndex((item) => item.id === above)
  if (index < 0) return [...tracks, track]
  return [...tracks.slice(0, index), track, ...tracks.slice(index)]
}

export type Placement = {
  name: string
  kind: TrackKind | null
  duration: number
  tone: Tone
  scene?: string
  assetId: string
}

const mediaScenes: Record<string, string> = {
  "ceremony-card": "ceremony",
  "vows-card": "vows",
  "portraits-card": "portraits",
  "dance-card": "dance",
  "rings-card": "rings",
  "exit-card": "exit",
}

const toolKind: Record<ToolId, TrackKind | null> = {
  media: "video",
  audio: "audio",
  text: "text",
  stickers: "sticker",
  effects: "effect",
  transitions: null,
  filters: "filter",
  adjustments: "adjust",
  templates: "text",
}

const toolTone: Record<ToolId, Tone> = {
  media: "picture",
  audio: "sound",
  text: "title",
  stickers: "sticker",
  effects: "effect",
  transitions: "picture",
  filters: "filter",
  adjustments: "adjust",
  templates: "title",
}

function clockSeconds(meta: string) {
  const match = meta.match(/(\d+):(\d{2})/)
  if (match) return Number(match[1]) * 60 + Number(match[2])
  const seconds = meta.match(/(\d+(?:\.\d+)?)s/)
  if (seconds) return Number(seconds[1])
  return null
}

export function placementFor(id: string): Placement | null {
  const stock = stockFootage.find((shot) => shot.id === id)
  if (stock) return { name: stock.name, kind: "video", duration: stock.duration, tone: "picture", scene: stock.id, assetId: id }
  const asset = assets.find((item) => item.id === id)
  if (!asset || asset.id === "wedding") return null
  const kind = toolKind[asset.tool]
  const duration = clockSeconds(asset.meta) ?? (kind === "text" ? 6 : kind === "sticker" ? 4 : 10)
  return {
    name: asset.name,
    kind,
    duration,
    tone: toolTone[asset.tool],
    scene: mediaScenes[asset.id],
    assetId: id,
  }
}

export function assetOnTimeline(state: ProjectState, assetId: string) {
  const source = placementFor(assetId)
  if (!source?.kind) return false
  return state.clips.some(
    (clip) => clip.assetId === assetId || clip.id === assetId || (source.scene !== undefined && clip.scene === source.scene),
  )
}

let draggingAssetId: string | null = null

export function beginAssetDrag(id: string) {
  draggingAssetId = id
}

export function assetDragId() {
  return draggingAssetId
}

export function endAssetDrag() {
  draggingAssetId = null
}

export type DropPlan = {
  ok: boolean
  note?: string
  start: number
  trackId: string | null
  insertAbove: string | null
  create: boolean
  hoverTrackId: string | null
  guide: number | null
}

function overlaps(clips: Clip[], trackId: string, start: number, duration: number, ignoreId?: string) {
  const end = start + duration
  return clips.some(
    (clip) =>
      clip.trackId === trackId &&
      clip.id !== ignoreId &&
      start < clip.start + clip.duration - 0.01 &&
      end > clip.start + 0.01,
  )
}

function snapHit(start: number, duration: number, points: number[], threshold: number) {
  const safe = Math.max(0, start)
  if (threshold <= 0) return { start: safe, guide: null as number | null }
  let best = safe
  let guide: number | null = null
  let bestDist = threshold
  for (const point of points) {
    const toStart = Math.abs(safe - point)
    if (toStart <= bestDist) {
      bestDist = toStart
      best = point
      guide = point
    }
    const toEnd = Math.abs(safe + duration - point)
    if (toEnd <= bestDist) {
      bestDist = toEnd
      best = Math.max(0, point - duration)
      guide = point
    }
  }
  return { start: best, guide }
}

function snapPoints(state: ProjectState, playhead: number, timelineEnd: number, ignoreId?: string) {
  const points = [0, Math.max(0, timelineEnd), playhead]
  for (const clip of state.clips) {
    if (clip.id === ignoreId) continue
    points.push(clip.start, clip.start + clip.duration)
  }
  for (const marker of state.markers) points.push(marker.time)
  if (state.guidesVisible) {
    for (const guide of state.guides) {
      if (guide.axis === "time") points.push(guide.at)
    }
  }
  const main = state.tracks.find((track) => track.main)
  if (!main) return points
  for (const transition of state.transitions) {
    const left = state.clips.find((clip) => clip.id === transition.leftId && clip.trackId === main.id)
    if (!left || left.id === ignoreId) continue
    const cut = left.start + left.duration
    const beside = state.clips.some(
      (clip) => clip.trackId === main.id && clip.id !== left.id && Math.abs(clip.start - cut) < 0.08,
    )
    if (beside) points.push(cut)
  }
  return points
}

export function adsorbTime(
  state: ProjectState,
  time: number,
  span: number,
  playhead: number,
  timelineEnd: number,
  threshold: number,
  ignoreId?: string,
) {
  return snapHit(time, span, snapPoints(state, playhead, timelineEnd, ignoreId), threshold).guide
}

function nextTrackId(tracks: Track[]) {
  let n = tracks.length + 1
  while (tracks.some((track) => track.id === `t${n}`)) n += 1
  return `t${n}`
}

function nextClipId(clips: Clip[]) {
  let n = clips.length + 1
  while (clips.some((clip) => clip.id === `c${n}`)) n += 1
  return `c${n}`
}

function insertTrackAbove(tracks: Track[], track: Track, aboveId: string) {
  const index = tracks.findIndex((item) => item.id === aboveId)
  if (index < 0) return placeTrack(tracks, track)
  return [...tracks.slice(0, index), track, ...tracks.slice(index)]
}

function packMain(clips: Clip[], mainId: string, movedId: string | null, dropTime: number) {
  const main = clips.filter((clip) => clip.trackId === mainId)
  const others = main.filter((clip) => clip.id !== movedId).sort((a, b) => a.start - b.start)
  const moved = movedId ? main.find((clip) => clip.id === movedId) : undefined
  let order = others
  if (moved) {
    let index = others.findIndex((clip) => dropTime < clip.start + clip.duration / 2)
    if (index < 0) index = others.length
    order = [...others.slice(0, index), moved, ...others.slice(index)]
  }
  let cursor = 0
  const starts = new Map<string, number>()
  for (const clip of order) {
    starts.set(clip.id, cursor)
    cursor += clip.duration
  }
  return clips.map((clip) => (starts.has(clip.id) ? { ...clip, start: starts.get(clip.id) ?? clip.start } : clip))
}

function clearEmpty(state: ProjectState): ProjectState {
  const used = new Set(state.clips.map((clip) => clip.trackId))
  let tracks = state.tracks.filter((track) => track.main || used.has(track.id))
  if (!tracks.some((track) => track.main)) {
    let id = "v1"
    while (tracks.some((track) => track.id === id)) id = `${id}m`
    tracks = placeTrack(tracks, { id, kind: "video", main: true })
  }
  return { ...state, tracks }
}

function refuse(start: number, hoverTrackId: string | null, note: string): DropPlan {
  return { ok: false, note, start, trackId: null, insertAbove: null, create: false, hoverTrackId, guide: null }
}

function planned(
  start: number,
  trackId: string | null,
  insertAbove: string | null,
  create: boolean,
  hoverTrackId: string | null,
  guide: number | null,
): DropPlan {
  return { ok: true, start, trackId, insertAbove, create, hoverTrackId, guide }
}

export function planAdd(
  state: ProjectState,
  source: Placement,
  time: number,
  hoverTrackId: string | null,
  playhead: number,
  threshold: number,
  timelineEnd: number,
): DropPlan {
  if (!source.kind) {
    return refuse(time, hoverTrackId, "Transitions sit between clips on the main track. They don't get a lane.")
  }
  const snapped = snapHit(time, source.duration, snapPoints(state, playhead, timelineEnd), threshold)
  const { start, guide } = snapped
  const hover = state.tracks.find((track) => track.id === hoverTrackId)
  if (!hover || hover.kind !== source.kind) {
    return planned(start, null, bandInsertAbove(state.tracks, source.kind, hover?.id ?? null), true, hover?.id ?? null, guide)
  }
  if (hover.locked) return refuse(start, hover.id, "This track is locked.")
  if (!(hover.main && state.magnet) && overlaps(state.clips, hover.id, start, source.duration)) {
    return refuse(start, hover.id, "That spot is taken.")
  }
  return planned(start, hover.id, null, false, hover.id, guide)
}

export function planMove(
  state: ProjectState,
  clipId: string,
  time: number,
  hoverTrackId: string | null,
  playhead: number,
  threshold: number,
  timelineEnd: number,
): DropPlan {
  const clip = state.clips.find((item) => item.id === clipId)
  const from = state.tracks.find((track) => track.id === clip?.trackId)
  if (!clip || !from) return refuse(time, hoverTrackId, "That clip is gone.")
  if (from.locked) return refuse(time, hoverTrackId, "This track is locked.")
  const snapped = snapHit(time, clip.duration, snapPoints(state, playhead, timelineEnd, clip.id), threshold)
  const { start, guide } = snapped
  const hover = hoverTrackId ? state.tracks.find((track) => track.id === hoverTrackId) : undefined
  if (!hover) {
    return planned(start, null, bandInsertAbove(state.tracks, from.kind, null), true, null, guide)
  }
  if (hover.kind !== from.kind) {
    return planned(start, null, bandInsertAbove(state.tracks, from.kind, hover.id), true, hover.id, guide)
  }
  const lands = (hover.main && state.magnet) || !overlaps(state.clips, hover.id, start, clip.duration, clip.id)
  if (lands && hover.locked) return refuse(start, hover.id, "This track is locked.")
  if (lands) return planned(start, hover.id, null, false, hover.id, guide)
  return planned(start, null, hover.id, true, hover.id, guide)
}

const MIN_CLIP = 0.2

const markerColors = ["#e4ff57", "#ff5f57", "#febc2e", "#00c8d7", "#c46a72"]

function rangesOverlap(start: number, end: number, otherStart: number, otherEnd: number) {
  return start < otherEnd - 0.05 && end > otherStart + 0.05
}

function exclusiveFollowers(clips: Clip[], mainClips: Clip[], owner: Clip) {
  const ownerEnd = owner.start + owner.duration
  return clips.filter((clip) => {
    if (clip.trackId === owner.trackId) return false
    if (!rangesOverlap(clip.start, clip.start + clip.duration, owner.start, ownerEnd)) return false
    return !mainClips.some(
      (other) =>
        other.id !== owner.id &&
        rangesOverlap(clip.start, clip.start + clip.duration, other.start, other.start + other.duration),
    )
  })
}

function shiftMarkers(before: Clip[], after: Clip[], markers: Marker[]) {
  return markers.flatMap((marker) => {
    if (!marker.clipId) return [marker]
    const previous = before.find((clip) => clip.id === marker.clipId)
    const next = after.find((clip) => clip.id === marker.clipId)
    if (!previous || !next) return []
    const media = previous.trimIn + (marker.time - previous.start)
    const time = next.start + (media - next.trimIn)
    if (time < next.start - 0.01 || time > next.start + next.duration + 0.01) return []
    return [{ ...marker, time }]
  })
}

function adjacentRight(clips: Clip[], mainId: string, left: Clip) {
  const cut = left.start + left.duration
  return clips.find(
    (clip) => clip.trackId === mainId && clip.id !== left.id && Math.abs(clip.start - cut) < 0.08,
  )
}

function clearTransitions(state: ProjectState): ProjectState {
  const main = state.tracks.find((track) => track.main)
  if (!main) return { ...state, transitions: [] }
  return {
    ...state,
    transitions: state.transitions.filter((transition) => {
      const left = state.clips.find((clip) => clip.id === transition.leftId && clip.trackId === main.id)
      return Boolean(left && adjacentRight(state.clips, main.id, left))
    }),
  }
}

function placeFollower(
  tracks: Track[],
  clips: Clip[],
  follower: Clip,
  start: number,
  duration: number,
) {
  const kind = tracks.find((track) => track.id === follower.trackId)?.kind ?? "audio"
  let nextTracks = tracks
  let trackId = follower.trackId
  if (overlaps(clips, trackId, start, duration, follower.id)) {
    trackId = nextTrackId(nextTracks)
    nextTracks = insertTrackAbove(nextTracks, { id: trackId, kind }, follower.trackId)
  }
  return {
    tracks: nextTracks,
    clips: clips.map((clip) => (clip.id === follower.id ? { ...clip, trackId, start, duration } : clip)),
  }
}

function applyLinkage(before: ProjectState, after: ProjectState): ProjectState {
  const main = after.tracks.find((track) => track.main)
  if (!main) return after
  const beforeMain = before.clips.filter((clip) => clip.trackId === main.id)
  let tracks = after.tracks
  let clips = after.clips
  for (const owner of beforeMain) {
    const next = clips.find((clip) => clip.id === owner.id && clip.trackId === main.id)
    const followers = exclusiveFollowers(before.clips, beforeMain, owner)
    if (!next) {
      const gone = new Set(followers.map((clip) => clip.id))
      clips = clips.filter((clip) => !gone.has(clip.id))
      continue
    }
    const delta = next.start - owner.start
    const grown = next.duration - owner.duration
    if (Math.abs(delta) < 0.001 && Math.abs(grown) < 0.001) continue
    for (const follower of followers) {
      const current = clips.find((clip) => clip.id === follower.id)
      if (!current) continue
      const flushEnd = Math.abs(follower.start + follower.duration - (owner.start + owner.duration)) < 0.15
      const start = Math.max(0, follower.start + delta)
      let end = start + follower.duration + (flushEnd ? grown : 0)
      const ownerEnd = next.start + next.duration
      if (end > ownerEnd) end = ownerEnd
      let duration = end - start
      if (duration < MIN_CLIP) {
        clips = clips.filter((clip) => clip.id !== current.id)
        continue
      }
      const maxDuration = Math.max(MIN_CLIP, current.source - current.trimIn)
      duration = Math.min(maxDuration, duration)
      const placed = placeFollower(tracks, clips, current, start, duration)
      tracks = placed.tracks
      clips = placed.clips
    }
  }
  return { ...after, tracks, clips }
}

function finish(before: ProjectState, after: ProjectState): ProjectState {
  const linked = before.linkage ? applyLinkage(before, after) : after
  return clearTransitions(
    clearEmpty({
      ...linked,
      markers: shiftMarkers(before.clips, linked.clips, before.markers),
    }),
  )
}

function land(state: ProjectState, clip: Clip, plan: DropPlan, kind: TrackKind): ProjectState {
  let tracks = state.tracks
  let trackId = plan.trackId
  if (plan.create || !trackId) {
    trackId = nextTrackId(tracks)
    const track = { id: trackId, kind }
    tracks = plan.insertAbove ? insertTrackAbove(tracks, track, plan.insertAbove) : placeTrack(tracks, track)
  }
  const placed = { ...clip, trackId, start: plan.start }
  const clips = state.clips.some((item) => item.id === clip.id)
    ? state.clips.map((item) => (item.id === clip.id ? placed : item))
    : [...state.clips, placed]
  const main = tracks.find((track) => track.main)
  const packed =
    state.magnet && main && trackId === main.id
      ? packMain(clips, main.id, clip.id, plan.start)
      : clips
  return finish(state, { ...state, tracks, clips: packed })
}

export function addPlacement(state: ProjectState, source: Placement, plan: DropPlan): { state: ProjectState; selectId?: string } {
  if (!plan.ok || !source.kind) return { state }
  const clip = bare({
    id: nextClipId(state.clips),
    trackId: plan.trackId ?? "",
    name: source.name,
    start: plan.start,
    duration: source.duration,
    tone: source.tone,
    scene: source.scene,
    assetId: source.assetId,
  })
  return { state: land(state, clip, plan, source.kind), selectId: clip.id }
}

export function insertPlacement(state: ProjectState, source: Placement): { state: ProjectState; note: string; selectId?: string; seek?: number } {
  if (!source.kind) return { state, note: "Transitions sit between clips on the main track." }
  const main = state.tracks.find((track) => track.main)
  const open = state.tracks.find((track) => track.kind === source.kind && !track.locked)
  const target = source.kind === "video" && main && !main.locked ? main : (open ?? null)
  const end = target
    ? state.clips.filter((clip) => clip.trackId === target.id).reduce((max, clip) => Math.max(max, clip.start + clip.duration), 0)
    : 0
  const plan = planAdd(state, source, end, target?.id ?? null, 0, 0, end)
  if (!plan.ok) return { state, note: plan.note ?? "That spot is taken." }
  const result = addPlacement(state, source, plan)
  return { state: result.state, note: `Added ${source.name}.`, selectId: result.selectId, seek: plan.start }
}

export function moveClip(state: ProjectState, clipId: string, plan: DropPlan): ProjectState {
  if (!plan.ok) return state
  const clip = state.clips.find((item) => item.id === clipId)
  const from = state.tracks.find((track) => track.id === clip?.trackId)
  if (!clip || !from) return state
  return land(state, clip, plan, from.kind)
}

export function nearestCut(state: ProjectState, time: number) {
  const main = state.tracks.find((track) => track.main)
  if (!main) return null
  const clips = state.clips.filter((clip) => clip.trackId === main.id).sort((a, b) => a.start - b.start)
  let best: { leftId: string; at: number; distance: number } | null = null
  for (let index = 0; index < clips.length - 1; index += 1) {
    const left = clips[index]
    const right = clips[index + 1]
    if (right.start - (left.start + left.duration) > 0.08) continue
    const at = left.start + left.duration
    const distance = Math.abs(time - at)
    if (!best || distance < best.distance) best = { leftId: left.id, at, distance }
  }
  if (!best || best.distance > 1.5) return null
  return { leftId: best.leftId, at: best.at }
}

export function addTransition(
  state: ProjectState,
  name: string,
  duration: number,
  time: number,
): { state: ProjectState; note: string; selectId?: string } {
  const cut = nearestCut(state, time)
  if (!cut) {
    return { state, note: "Transitions sit between clips on the main track. They don't get a lane." }
  }
  const existing = state.transitions.find((transition) => transition.leftId === cut.leftId)
  let count = state.transitions.length + 1
  while (state.transitions.some((transition) => transition.id === `tr${count}`)) count += 1
  const id = existing?.id ?? `tr${count}`
  const transition = { id, name, leftId: cut.leftId, duration }
  const left = state.clips.find((clip) => clip.id === cut.leftId)
  return {
    state: {
      ...state,
      transitions: existing
        ? state.transitions.map((item) => (item.id === existing.id ? transition : item))
        : [...state.transitions, transition],
    },
    note: `${name} sits on the cut after ${left?.name ?? "the clip"}.`,
    selectId: id,
  }
}

function transitionCovers(state: ProjectState, time: number) {
  const main = state.tracks.find((track) => track.main)
  if (!main) return false
  return state.transitions.some((transition) => {
    const left = state.clips.find((clip) => clip.id === transition.leftId && clip.trackId === main.id)
    if (!left || !adjacentRight(state.clips, main.id, left)) return false
    const cut = left.start + left.duration
    return time >= cut - transition.duration / 2 && time <= cut + transition.duration / 2
  })
}

export function splitClip(state: ProjectState, clipId: string, time: number): { state: ProjectState; selectId?: string; note?: string } {
  const clip = state.clips.find((item) => item.id === clipId)
  if (!clip) return { state }
  if (transitionCovers(state, time)) return { state, note: "Transitions cannot be split." }
  if (time <= clip.start + MIN_CLIP || time >= clip.start + clip.duration - MIN_CLIP) {
    return { state, note: "This position cannot be split." }
  }
  const rightId = nextClipId(state.clips)
  const leftDuration = time - clip.start
  const left = { ...clip, duration: leftDuration }
  const right = {
    ...clip,
    id: rightId,
    start: time,
    duration: clip.duration - leftDuration,
    trimIn: clip.trimIn + leftDuration,
  }
  const markers = state.markers.map((marker) =>
    marker.clipId === clip.id && marker.time >= time ? { ...marker, clipId: rightId } : marker,
  )
  const transitions = state.transitions.map((transition) =>
    transition.leftId === clip.id ? { ...transition, leftId: rightId } : transition,
  )
  return {
    state: {
      ...state,
      markers,
      transitions,
      clips: state.clips.map((item) => (item.id === clip.id ? left : item)).concat(right),
    },
    selectId: rightId,
  }
}

function snapEdge(value: number, points: number[], threshold: number) {
  if (threshold <= 0) return value
  let best = value
  let bestDist = threshold
  for (const point of points) {
    const distance = Math.abs(value - point)
    if (distance <= bestDist) {
      bestDist = distance
      best = point
    }
  }
  return best
}

export function trimClip(
  state: ProjectState,
  clipId: string,
  edge: "in" | "out",
  time: number,
  playhead: number,
  threshold: number,
  timelineEnd: number,
): ProjectState {
  const clip = state.clips.find((item) => item.id === clipId)
  const track = state.tracks.find((item) => item.id === clip?.trackId)
  if (!clip || !track || track.locked) return state
  const magnetic = Boolean(track.main && state.magnet)
  const others = state.clips
    .filter((item) => item.trackId === clip.trackId && item.id !== clip.id)
    .sort((a, b) => a.start - b.start)
  const previous = [...others].reverse().find((item) => item.start + item.duration <= clip.start + 0.05)
  const next = others.find((item) => item.start >= clip.start + clip.duration - 0.05)
  const points = snapPoints(state, playhead, timelineEnd, clip.id)
  let start = clip.start
  let duration = clip.duration
  let trimIn = clip.trimIn
  if (edge === "in") {
    if (magnetic) {
      let delta = snapEdge(time, points, threshold) - clip.start
      delta = Math.max(-clip.trimIn, Math.min(clip.duration - MIN_CLIP, delta))
      trimIn = clip.trimIn + delta
      duration = clip.duration - delta
    } else {
      const end = clip.start + clip.duration
      const minStart = Math.max(0, previous ? previous.start + previous.duration : 0, clip.start - clip.trimIn)
      const maxStart = end - MIN_CLIP
      start = Math.min(maxStart, Math.max(minStart, snapEdge(time, points, threshold)))
      trimIn = clip.trimIn + (start - clip.start)
      duration = end - start
    }
  } else {
    const minEnd = clip.start + MIN_CLIP
    const maxEnd = clip.start + (clip.source - clip.trimIn)
    let end = Math.min(maxEnd, Math.max(minEnd, snapEdge(time, points, threshold)))
    if (!magnetic && next && end > next.start) end = Math.max(minEnd, next.start)
    if (end < minEnd) return state
    duration = end - clip.start
  }
  if (
    Math.abs(start - clip.start) < 0.001 &&
    Math.abs(duration - clip.duration) < 0.001 &&
    Math.abs(trimIn - clip.trimIn) < 0.001
  ) {
    return state
  }
  let clips = state.clips.map((item) => (item.id === clip.id ? { ...item, start, duration, trimIn } : item))
  if (magnetic) clips = packMain(clips, track.id, null, 0)
  return finish(state, { ...state, clips })
}

export function setMagnet(state: ProjectState, magnet: boolean): ProjectState {
  if (state.tracks.some((track) => track.main && track.locked)) return state
  if (state.magnet === magnet) return state
  if (!magnet) return { ...state, magnet }
  const main = state.tracks.find((track) => track.main)
  if (!main) return { ...state, magnet: true }
  return finish(state, { ...state, magnet: true, clips: packMain(state.clips, main.id, null, 0) })
}

export function setLinkage(state: ProjectState, linkage: boolean): ProjectState {
  if (state.linkage === linkage) return state
  return { ...state, linkage }
}

export function addMarker(
  state: ProjectState,
  time: number,
  clipId: string | null,
  beat: boolean,
): { state: ProjectState; note?: string } {
  const clip = clipId ? state.clips.find((item) => item.id === clipId) : undefined
  if (beat) {
    const track = state.tracks.find((item) => item.id === clip?.trackId)
    if (!clip || track?.kind !== "audio") return { state, note: "Beat marks sit on audio." }
  }
  if (clip && (time < clip.start || time > clip.start + clip.duration)) return { state }
  const color = beat ? "#ffffff" : markerColors[state.markers.filter((marker) => !marker.beat).length % markerColors.length]
  let count = state.markers.length + 1
  while (state.markers.some((marker) => marker.id === `m${count}`)) count += 1
  const marker: Marker = { id: `m${count}`, time, color, beat, clipId }
  return { state: { ...state, markers: [...state.markers, marker] } }
}

export function removeMarker(state: ProjectState, markerId: string): ProjectState {
  return { ...state, markers: state.markers.filter((marker) => marker.id !== markerId) }
}

export function addGuide(state: ProjectState, axis: Guide["axis"], at: number): { state: ProjectState; note?: string; selectId: string } {
  if (state.guidesLocked) return { state, note: "Reference lines are locked.", selectId: state.guides[0]?.id ?? "" }
  let count = state.guides.length + 1
  while (state.guides.some((guide) => guide.id === `g${count}`)) count += 1
  const guide: Guide = { id: `g${count}`, axis, at }
  return { state: { ...state, guides: [...state.guides, guide], guidesVisible: true }, selectId: guide.id }
}

export function moveGuide(state: ProjectState, guideId: string, at: number): ProjectState {
  if (state.guidesLocked) return state
  return { ...state, guides: state.guides.map((guide) => (guide.id === guideId ? { ...guide, at } : guide)) }
}

export function removeGuide(state: ProjectState, guideId: string): ProjectState {
  if (state.guidesLocked) return state
  return { ...state, guides: state.guides.filter((guide) => guide.id !== guideId) }
}

export function setGuidesVisible(state: ProjectState, visible: boolean): ProjectState {
  return { ...state, guidesVisible: visible }
}

export function setGuidesLocked(state: ProjectState, locked: boolean): ProjectState {
  return { ...state, guidesLocked: locked }
}

export function setTrackLocked(state: ProjectState, trackId: string, locked: boolean): ProjectState {
  return {
    ...state,
    tracks: state.tracks.map((track) => (track.id === trackId ? { ...track, locked } : track)),
  }
}

export function deleteSelection(state: ProjectState, selectedIds: string[]): { state: ProjectState; note?: string; selectIds: string[] } {
  if (selectedIds.length <= 1) return deleteOne(state, selectedIds[0] ?? "")
  const wanted = selectedIds
    .map((id) => state.clips.find((clip) => clip.id === id))
    .filter((clip): clip is Clip => Boolean(clip))
  if (wanted.length <= 1) return deleteOne(state, selectedIds[0] ?? "")
  const removing = wanted.filter((clip) => {
    const track = state.tracks.find((item) => item.id === clip.trackId)
    return !track?.locked
  })
  const lockedSkipped = removing.length < wanted.length
  const main = state.tracks.find((track) => track.main)
  if (removing.length === 0) {
    return { state, note: lockedSkipped ? "This track is locked." : undefined, selectIds: selectedIds }
  }
  const gone = new Set(removing.map((clip) => clip.id))
  let clips = state.clips.filter((clip) => !gone.has(clip.id))
  if (state.magnet && main && removing.some((clip) => clip.trackId === main.id)) clips = packMain(clips, main.id, null, 0)
  const next = finish(state, { ...state, clips })
  const followed = state.linkage && next.clips.length < state.clips.length - removing.length
  const selectIds = selectedIds.filter((id) => !gone.has(id))
  return {
    state: next,
    note: followed ? `Removed ${removing.length} clips. Linked clips followed.` : undefined,
    selectIds,
  }
}

function deleteOne(state: ProjectState, selectedId: string): { state: ProjectState; note?: string; selectIds: string[] } {
  const guide = state.guides.find((item) => item.id === selectedId)
  if (guide) {
    if (state.guidesLocked) return { state, note: "Reference lines are locked.", selectIds: [guide.id] }
    return { state: removeGuide(state, guide.id), selectIds: [] }
  }
  const transition = state.transitions.find((item) => item.id === selectedId)
  if (transition) {
    return {
      state: { ...state, transitions: state.transitions.filter((item) => item.id !== transition.id) },
      selectIds: [],
    }
  }
  const clip = state.clips.find((item) => item.id === selectedId)
  if (!clip) return { state, selectIds: selectedId ? [selectedId] : [] }
  const from = state.tracks.find((track) => track.id === clip.trackId)
  if (from?.locked) return { state, note: "This track is locked.", selectIds: [clip.id] }
  let clips = state.clips.filter((item) => item.id !== clip.id)
  if (state.magnet && from?.main) clips = packMain(clips, from.id, null, 0)
  const next = finish(state, { ...state, clips })
  const followed = state.linkage && next.clips.length < state.clips.length - 1
  return {
    state: next,
    note: followed ? `Removed ${clip.name}. Linked clips followed.` : undefined,
    selectIds: [],
  }
}

function withClip(clips: Clip[], clip: Clip) {
  if (clips.some((item) => item.id === clip.id)) return clips
  return [...clips, clip]
}

export function applyCommand(state: ProjectState, raw: string): CommandResult {
  const command = raw.trim().toLowerCase()

  if (command.includes("silence")) {
    if (state.silencesRemoved) {
      return { state, note: "Silences are already out of this cut." }
    }
    return {
      state: { ...state, silencesRemoved: true },
      note: "Removed 11 silent gaps from Ceremony and Vows.",
    }
  }

  if (command.includes("wedding") || command.includes("template")) {
    const already = state.clips.some((clip) => clip.id === "lower") && state.clips.some((clip) => clip.id === "end")
    if (already) {
      return { state, note: "The wedding film is already on the timeline.", selectId: "lower", seek: 22 }
    }
    const text = state.tracks.find((track) => track.kind === "text")
    const trackId = text?.id ?? "v2"
    const tracks = text ? state.tracks : placeTrack(state.tracks, { id: trackId, kind: "text" })
    return {
      state: {
        ...state,
        tracks,
        clips: withClip(
          withClip(
            state.clips,
            bare({
              id: "lower",
              trackId,
              name: "Amina  ·  Jules",
              start: 22,
              duration: 8,
              tone: "title",
            }),
          ),
          bare({
            id: "end",
            trackId,
            name: "12 June",
            start: 84,
            duration: 8,
            tone: "title",
          }),
        ),
      },
      note: "Lower third and end card are on a text track.",
      selectId: "lower",
      seek: 22,
    }
  }

  if (command.startsWith("stock ")) {
    const id = command.slice("stock ".length).trim()
    const shot = stockFootage.find((item) => item.id === id)
    if (!shot) return { state, note: "That stock shot is not in the free shelf." }
    if (state.clips.some((clip) => clip.id === shot.id)) {
      return { state, note: `${shot.name} is already on the timeline.`, selectId: shot.id }
    }
    const tracks = placeTrack(state.tracks, { id: "s1", kind: "video" })
    const start = state.clips
      .filter((clip) => clip.trackId === "s1")
      .reduce((end, clip) => Math.max(end, clip.start + clip.duration), 0)
    return {
      state: {
        ...state,
        tracks,
        clips: withClip(
          state.clips,
          bare({
            id: shot.id,
            trackId: "s1",
            name: shot.name,
            start,
            duration: shot.duration,
            tone: "picture",
            scene: shot.id,
          }),
        ),
      },
      note: `Added free stock “${shot.name}” above the main video.`,
      selectId: shot.id,
      seek: start,
    }
  }

  if (command.includes("generat") || command.includes("insert") || command.includes("0:32")) {
    if (state.clips.some((clip) => clip.id === "gen-shot")) {
      return {
        state,
        note: "A generated shot is already sitting at 0:32.",
        selectId: "gen-shot",
        seek: 32,
      }
    }
    const tracks = placeTrack(state.tracks, { id: "g1", kind: "video" })
    return {
      state: {
        ...state,
        tracks,
        clips: [
          ...state.clips,
          bare({
            id: "gen-shot",
            trackId: "g1",
            name: "Generated shot",
            start: 32,
            duration: 6,
            tone: "generated",
          }),
        ],
      },
      note: "Generated a shot and inserted it at 0:32.",
      selectId: "gen-shot",
      seek: 32,
    }
  }

  return {
    state,
    note: "Commands become timeline edits. Try silences, a shot at 0:32, or a wedding film.",
  }
}
