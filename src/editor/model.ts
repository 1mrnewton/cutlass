export const DURATION = 92

export type ToolId =
  | "media"
  | "audio"
  | "text"
  | "stickers"
  | "effects"
  | "transitions"
  | "templates"

export type Tone = "picture" | "title" | "sound" | "generated"

export type Track = {
  id: string
  label: string
  kind: "video" | "text" | "audio"
}

export type Clip = {
  id: string
  trackId: string
  name: string
  start: number
  duration: number
  tone: Tone
}

export type ProjectState = {
  tracks: Track[]
  clips: Clip[]
  silencesRemoved: boolean
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
  tool: ToolId
  note: string | null
  onTogglePlay: () => void
  onSeek: (time: number) => void
  onSelect: (id: string) => void
  onTool: (tool: ToolId) => void
  onCommand: (command: string) => void
  onExport: () => void
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
  { id: "templates", label: "Templates" },
]

export const suggestions = [
  { label: "Remove silences", command: "Remove all silences" },
  { label: "Insert shot at 0:32", command: "Generate a shot and insert it at 0:32" },
  { label: "Wedding film", command: "Using all these footages, make a wedding film" },
]

export const rulerMarks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

export const scenes: Record<string, [string, string, string]> = {
  ceremony: ["#2c1b12", "#e09a62", "#100e0c"],
  vows: ["#261820", "#c46a72", "#110c10"],
  portraits: ["#172430", "#8aa4b8", "#0c1014"],
  dance: ["#2a2012", "#e2a45c", "#12100c"],
}

export const initialProject: ProjectState = {
  silencesRemoved: false,
  tracks: [
    { id: "v2", label: "Titles", kind: "text" },
    { id: "v1", label: "Video", kind: "video" },
    { id: "a2", label: "Talk", kind: "audio" },
    { id: "a1", label: "Music", kind: "audio" },
  ],
  clips: [
    { id: "ceremony", trackId: "v1", name: "Ceremony", start: 0, duration: 18, tone: "picture" },
    { id: "vows", trackId: "v1", name: "Vows", start: 18, duration: 30, tone: "picture" },
    { id: "portraits", trackId: "v1", name: "Portraits", start: 48, duration: 20, tone: "picture" },
    { id: "dance", trackId: "v1", name: "Dance", start: 68, duration: 24, tone: "picture" },
    { id: "title", trackId: "v2", name: "Amina & Jules", start: 3, duration: 7, tone: "title" },
    { id: "vows-mic", trackId: "a2", name: "Vows mic", start: 20, duration: 26, tone: "sound" },
    { id: "music", trackId: "a1", name: "Paper boat", start: 0, duration: 92, tone: "sound" },
  ],
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
  { id: "soft", tool: "effects", name: "Soft light", meta: "Look", swatch: "#f0d8b0" },
  { id: "grain", tool: "effects", name: "Fade grain", meta: "Texture", swatch: "#b7aa9a" },
  { id: "bloom", tool: "effects", name: "Bloom", meta: "Light", swatch: "#f4f0c8" },
  { id: "dip", tool: "transitions", name: "Dip to black", meta: "0.4s", swatch: "#2a2c30" },
  { id: "whip", tool: "transitions", name: "Whip", meta: "0.2s", swatch: "#d7d2c6" },
  { id: "dissolve", tool: "transitions", name: "Dissolve", meta: "0.6s", swatch: "#8d8a84" },
  { id: "leak", tool: "transitions", name: "Light leak", meta: "0.5s", swatch: "#e7a15a" },
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

export function formatMark(seconds: number) {
  const minutes = Math.floor(seconds / 60)
  const whole = Math.floor(seconds - minutes * 60)
  return `${minutes}:${whole.toString().padStart(2, "0")}`
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
      return { state, note: "The wedding film is already on the timeline.", selectId: "title", seek: 3 }
    }
    return {
      state: {
        ...state,
        clips: withClip(
          withClip(state.clips, {
            id: "lower",
            trackId: "v2",
            name: "Amina  ·  Jules",
            start: 22,
            duration: 8,
            tone: "title",
          }),
          {
            id: "end",
            trackId: "v2",
            name: "12 June",
            start: 84,
            duration: 8,
            tone: "title",
          },
        ),
      },
      note: "Cut a wedding film from the four shots. Title, lower third, and end card are on the timeline.",
      selectId: "title",
      seek: 3,
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
    const tracks = state.tracks.some((track) => track.id === "g1")
      ? state.tracks
      : [{ id: "g1", label: "Gen", kind: "video" as const }, ...state.tracks]
    return {
      state: {
        ...state,
        tracks,
        clips: [
          ...state.clips,
          {
            id: "gen-shot",
            trackId: "g1",
            name: "Generated shot",
            start: 32,
            duration: 6,
            tone: "generated",
          },
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
