import { useState, type ReactNode } from "react"
import { ColorTools } from "./ColorTools"
import { Icon } from "./icons"
import { useLook, type Look } from "./look"
import type { Clip } from "./model"

type VideoTab = "video" | "audio" | "speed" | "animation" | "adjust" | "style"
type VideoPage = "basic" | "cutout" | "mask" | "enhance"
type AudioPage = "basic" | "voice"
type AnimSlot = "in" | "out" | "combo"
type AdjustPage = "basic" | "hsl" | "curves" | "wheel" | "mask"

const videoTabs: { id: VideoTab; label: string }[] = [
  { id: "video", label: "Video" },
  { id: "audio", label: "Audio" },
  { id: "speed", label: "Speed" },
  { id: "animation", label: "Animation" },
  { id: "adjust", label: "Adjust" },
  { id: "style", label: "AI stylize" },
]

const videoFeatures: { id: string; label: string; badge?: string; apply?: boolean }[] = [
  { id: "blend", label: "Blend" },
  { id: "stabilize", label: "Stabilize" },
  { id: "enhance", label: "Enhance quality" },
  { id: "noise", label: "Noise reduction" },
  { id: "optical", label: "Optical flow", apply: true },
  { id: "remove", label: "AI remove" },
  { id: "expand", label: "AI expand", badge: "Free" },
  { id: "remix", label: "AI remix" },
  { id: "lipsync", label: "Lip sync" },
  { id: "relight", label: "Relight" },
  { id: "reframe", label: "Auto reframe" },
  { id: "flicker", label: "Removing video flickers" },
  { id: "movement", label: "AI movement" },
  { id: "tracking", label: "Camera tracking" },
  { id: "blur", label: "Motion blur" },
  { id: "canvas", label: "Canvas", apply: true },
]

const audioFeatures: { id: string; label: string; badge?: string; note?: string }[] = [
  { id: "loudness", label: "Loudness normalization", note: "Normalizes the selected clip to a target level." },
  { id: "enhance-voice", label: "Enhance voice" },
  { id: "translator", label: "Video translator" },
  { id: "reduce", label: "Reduce noise" },
  { id: "isolate", label: "Isolate voice", badge: "Free" },
  { id: "fill", label: "Fill channel" },
]

const blendModes = ["Normal", "Multiply", "Screen", "Overlay", "Soft light"]

const adjustPages: { id: AdjustPage; label: string }[] = [
  { id: "basic", label: "Basic" },
  { id: "hsl", label: "HSL" },
  { id: "curves", label: "Curves" },
  { id: "wheel", label: "Color wheel" },
  { id: "mask", label: "Mask" },
]

const lutNames = ["None", "Warm", "Cool", "Film", "Mono"]

const gradeGroups: { name: string; sliders: { id: string; label: string; min: number; max: number; grade: string }[] }[] = [
  {
    name: "Color",
    sliders: [
      { id: "temp", label: "Temp", min: -100, max: 100, grade: "linear-gradient(90deg, #2f6fed, #d9d9d9 50%, #f2d14a)" },
      { id: "hue", label: "Hue", min: -100, max: 100, grade: "linear-gradient(90deg, #3aaa4a, #d0d0d0 50%, #b44ad4)" },
      { id: "saturation", label: "Saturation", min: -100, max: 100, grade: "linear-gradient(90deg, #6a3030, #e24b4b)" },
    ],
  },
  {
    name: "Lightness",
    sliders: [
      { id: "brightness", label: "Brightness", min: -100, max: 100, grade: "linear-gradient(90deg, #3a3a3a, #f2f2f2)" },
      { id: "contrast", label: "Contrast", min: -100, max: 100, grade: "linear-gradient(90deg, #6a6a6a, #111 50%, #f5f5f5)" },
      { id: "highlight", label: "Highlight", min: -100, max: 100, grade: "linear-gradient(90deg, #8a8a8a, #fff)" },
      { id: "shadow", label: "Shadow", min: -100, max: 100, grade: "linear-gradient(90deg, #111, #9a9a9a)" },
      { id: "light", label: "Light", min: -100, max: 100, grade: "linear-gradient(90deg, #3a3a3a, #fff6d0)" },
      { id: "fade", label: "Fade", min: 0, max: 100, grade: "linear-gradient(90deg, #2a2a2a, #cfcfcf)" },
    ],
  },
  {
    name: "Effects",
    sliders: [
      { id: "sharpen", label: "Sharpen", min: 0, max: 100, grade: "linear-gradient(90deg, #2a2a2a, #eee)" },
      { id: "vignette", label: "Vignette", min: 0, max: 100, grade: "linear-gradient(90deg, #2a2a2a, #111)" },
      { id: "particle", label: "Particle", min: 0, max: 100, grade: "linear-gradient(90deg, #2a2a2a, #d8d8d8)" },
    ],
  },
]

const animSlots: { id: AnimSlot; label: string }[] = [
  { id: "in", label: "In" },
  { id: "out", label: "Out" },
  { id: "combo", label: "Combo" },
]

const animations: Record<AnimSlot, { id: string; label: string; look: string }[]> = {
  in: [
    { id: "fade-in", label: "Fade in", look: "fade" },
    { id: "slide-up", label: "Slide up", look: "up" },
    { id: "slide-down", label: "Slide down", look: "down" },
    { id: "slide-left", label: "Slide left", look: "left" },
    { id: "slide-right", label: "Slide right", look: "right" },
    { id: "zoom-in", label: "Zoom in", look: "zoom" },
    { id: "pop", label: "Pop", look: "pop" },
    { id: "spin", label: "Spin", look: "spin" },
    { id: "wipe", label: "Wipe", look: "wipe" },
    { id: "blur-in", label: "Blur in", look: "blur" },
    { id: "rise", label: "Rise", look: "rise" },
    { id: "flip", label: "Flip", look: "flip" },
  ],
  out: [
    { id: "fade-out", label: "Fade out", look: "fade" },
    { id: "slide-up-out", label: "Slide up", look: "up" },
    { id: "slide-down-out", label: "Slide down", look: "down" },
    { id: "slide-left-out", label: "Slide left", look: "left" },
    { id: "slide-right-out", label: "Slide right", look: "right" },
    { id: "zoom-out", label: "Zoom out", look: "zoom" },
    { id: "shrink", label: "Shrink", look: "pop" },
    { id: "spin-out", label: "Spin", look: "spin" },
    { id: "wipe-out", label: "Wipe", look: "wipe" },
    { id: "blur-out", label: "Blur out", look: "blur" },
    { id: "fall", label: "Fall", look: "down" },
    { id: "flip-out", label: "Flip", look: "flip" },
  ],
  combo: [
    { id: "shake", label: "Shake", look: "shake" },
    { id: "pulse", label: "Pulse", look: "pulse" },
    { id: "float", label: "Float", look: "float" },
    { id: "bounce", label: "Bounce", look: "bounce" },
    { id: "drift", label: "Drift", look: "right" },
    { id: "flicker", label: "Flicker", look: "flicker" },
    { id: "swing", label: "Swing", look: "spin" },
    { id: "breathe", label: "Breathe", look: "pulse" },
  ],
}

export function VideoInspector({
  clip,
  selectedCount,
  width,
  onNote,
}: {
  clip: Clip
  selectedCount: number
  width: number
  onNote: (note: string) => void
}) {
  const [tab, setTab] = useState<VideoTab>("video")
  const [videoPage, setVideoPage] = useState<VideoPage>("basic")
  const [audioPage, setAudioPage] = useState<AudioPage>("basic")
  const [positionOpen, setPositionOpen] = useState(true)
  const [audioOpen, setAudioOpen] = useState(true)
  const [openFeature, setOpenFeature] = useState<string | null>(null)
  const [enabled, setEnabled] = useState<Record<string, boolean>>({})
  const [scale, setScale] = useState(100)
  const [uniform, setUniform] = useState(true)
  const [x, setX] = useState(0)
  const [y, setY] = useState(0)
  const [rotate, setRotate] = useState(0)
  const [volume, setVolume] = useState(0)
  const [fadeIn, setFadeIn] = useState(0)
  const [fadeOut, setFadeOut] = useState(0)
  const [audioOn, setAudioOn] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [blend, setBlend] = useState("Normal")
  const [amount, setAmount] = useState<Record<string, number>>({})
  const [animSlot, setAnimSlot] = useState<AnimSlot>("in")
  const [animQuery, setAnimQuery] = useState("")
  const [animPick, setAnimPick] = useState<Record<AnimSlot, string | null>>({ in: null, out: null, combo: null })
  const [animDuration, setAnimDuration] = useState<Record<AnimSlot, number>>({ in: 2, out: 2, combo: 2 })
  const [adjustPage, setAdjustPage] = useState<AdjustPage>("basic")
  const [adjustOn, setAdjustOn] = useState<Record<string, boolean>>({ lut: true, grade: true })
  const [adjustOpen, setAdjustOpen] = useState<Record<string, boolean>>({ lut: true, grade: true })
  const [lut, setLut] = useState("None")
  const [lutStrength, setLutStrength] = useState(100)
  const [skin, setSkin] = useState(true)
  const [grades, setGrades] = useState<Record<string, number>>({})
  const { setLook } = useLook()

  const toggleFeature = (id: string) => {
    setEnabled((current) => ({ ...current, [id]: !current[id] }))
    setOpenFeature(id)
  }

  return (
    <aside className="inspector clip-inspector" style={{ width }} aria-label={`${clip.name} inspector`}>
      <div className="clip-tabs" role="tablist" aria-label="Clip inspector">
        {videoTabs.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={tab === item.id}
            data-active={tab === item.id ? "true" : "false"}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      {selectedCount > 1 && <p className="clip-count">{selectedCount} clips selected.</p>}
      {tab === "video" && (
        <div className="clip-pages" role="tablist" aria-label="Video">
          {(
            [
              ["basic", "Basic"],
              ["cutout", "Cutout"],
              ["mask", "Mask"],
              ["enhance", "Enhance"],
            ] as const
          ).map(([id, label]) => (
            <button key={id} type="button" role="tab" aria-selected={videoPage === id} data-active={videoPage === id ? "true" : "false"} onClick={() => setVideoPage(id)}>
              {label}
            </button>
          ))}
        </div>
      )}
      {tab === "audio" && (
        <div className="clip-pages" role="tablist" aria-label="Audio">
          {(
            [
              ["basic", "Basic"],
              ["voice", "Voice changer"],
            ] as const
          ).map(([id, label]) => (
            <button key={id} type="button" role="tab" aria-selected={audioPage === id} data-active={audioPage === id ? "true" : "false"} onClick={() => setAudioPage(id)}>
              {label}
            </button>
          ))}
        </div>
      )}
      {tab === "animation" ? (
        <AnimationTab
          slot={animSlot}
          query={animQuery}
          pick={animPick[animSlot]}
          duration={animDuration[animSlot]}
          onSlot={setAnimSlot}
          onQuery={setAnimQuery}
          onPick={(id) => setAnimPick((current) => ({ ...current, [animSlot]: id }))}
          onDuration={(value) => setAnimDuration((current) => ({ ...current, [animSlot]: value }))}
        />
      ) : tab === "adjust" ? (
        <AdjustTab
          page={adjustPage}
          on={adjustOn}
          open={adjustOpen}
          lut={lut}
          strength={lutStrength}
          skin={skin}
          grades={grades}
          onPage={setAdjustPage}
          onToggle={(id) => {
            setAdjustOn((current) => ({ ...current, [id]: !current[id] }))
            setAdjustOpen((current) => ({ ...current, [id]: true }))
          }}
          onOpen={(id) => setAdjustOpen((current) => ({ ...current, [id]: !current[id] }))}
          onLut={setLut}
          onStrength={setLutStrength}
          onSkin={setSkin}
          onGrade={(id, value) => {
            setGrades((current) => ({ ...current, [id]: value }))
            if (id === "temp" || id === "saturation" || id === "brightness") {
              setLook(clip.id, { temp: id === "temp" ? value : grades.temp, sat: id === "saturation" ? value : grades.saturation, bright: id === "brightness" ? value : grades.brightness })
            }
          }}
          onResetLut={() => { setLut("None"); setLutStrength(100); setSkin(true) }}
          onResetGrade={() => {
            setGrades({})
            setLook(clip.id, { temp: 0, sat: 0, bright: 0 })
          }}
          onTint={(patch) => setLook(clip.id, patch)}
          onNote={onNote}
        />
      ) : (
      <div className="clip-body">
        {tab === "video" && videoPage === "basic" && (
          <>
            <section className="clip-section">
              <header>
                <button type="button" className="section-name" aria-expanded={positionOpen} onClick={() => setPositionOpen((value) => !value)}>
                  <Icon name="chevron" size={14} />
                  Position & Size
                </button>
                <span className="key-tools">
                  <button type="button" aria-label="Reset position and size" onClick={() => { setScale(100); setUniform(true); setX(0); setY(0); setRotate(0) }}>
                    <Icon name="reset" size={14} />
                  </button>
                  <button type="button" aria-label="Previous keyframe" onClick={() => onNote("No keyframes on this clip.")}>
                    <Icon name="chevronLeft" size={14} />
                  </button>
                  <button type="button" aria-label="Add keyframe" onClick={() => onNote("The prototype does not store keyframes.")}>
                    <Icon name="keyframe" size={14} />
                  </button>
                  <button type="button" aria-label="Next keyframe" onClick={() => onNote("No keyframes on this clip.")}>
                    <Icon name="chevron" size={14} />
                  </button>
                </span>
              </header>
              {positionOpen && (
                <div className="section-body">
                  <label className="slider-row">
                    <span>Scale</span>
                    <input type="range" min={1} max={500} value={scale} aria-label="Scale" onChange={(event) => setScale(Number(event.target.value))} />
                    <output>{scale}%</output>
                  </label>
                  <div className="switch-row">
                    <span>Uniform scale</span>
                    <button type="button" className="switch" aria-pressed={uniform} aria-label={uniform ? "Uniform scale on" : "Uniform scale off"} onClick={() => setUniform((value) => !value)}>
                      <span />
                    </button>
                  </div>
                  <div className="pair-row">
                    <span>Position</span>
                    <label>
                      X
                      <input type="number" value={x} aria-label="Position X" onChange={(event) => setX(Number(event.target.value))} />
                    </label>
                    <label>
                      Y
                      <input type="number" value={y} aria-label="Position Y" onChange={(event) => setY(Number(event.target.value))} />
                    </label>
                  </div>
                  <div className="pair-row">
                    <span>Rotate</span>
                    <label>
                      <Icon name="rotate" size={14} />
                      <input type="number" value={rotate} step={0.1} aria-label="Rotate" onChange={(event) => setRotate(Number(event.target.value))} />
                      <span>°</span>
                    </label>
                  </div>
                  <div className="align-bar" role="toolbar" aria-label="Align">
                    {(
                      [
                        ["alignLeft", "Align left"],
                        ["alignCenter", "Align center"],
                        ["alignRight", "Align right"],
                        ["alignTop", "Align top"],
                        ["alignMiddle", "Align middle"],
                        ["alignBottom", "Align bottom"],
                      ] as const
                    ).map(([name, label]) => (
                      <button
                        key={name}
                        type="button"
                        aria-label={label}
                        onClick={() => {
                          if (name === "alignLeft") setX(-960)
                          if (name === "alignCenter" || name === "alignMiddle") {
                            if (name === "alignCenter") setX(0)
                            else setY(0)
                          }
                          if (name === "alignRight") setX(960)
                          if (name === "alignTop") setY(-540)
                          if (name === "alignBottom") setY(540)
                        }}
                      >
                        <Icon name={name} size={16} />
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </section>
            {videoFeatures.map((feature) => (
              <FeatureRow
                key={feature.id}
                label={feature.label}
                badge={feature.badge}
                apply={feature.apply}
                open={openFeature === feature.id}
                enabled={enabled[feature.id] ?? false}
                onToggle={() => toggleFeature(feature.id)}
                onOpen={() => setOpenFeature((current) => (current === feature.id ? null : feature.id))}
                onApply={() => onNote("The prototype does not change the other clips.")}
              >
                {feature.id === "blend" ? (
                  <label className="blend-row">
                    Mode
                    <select value={blend} aria-label="Blend mode" onChange={(event) => setBlend(event.target.value)}>
                      {blendModes.map((mode) => (
                        <option key={mode}>{mode}</option>
                      ))}
                    </select>
                  </label>
                ) : (
                  <label className="slider-row">
                    <span>Amount</span>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={amount[feature.id] ?? 0}
                      aria-label={`${feature.label} amount`}
                      onChange={(event) => setAmount((current) => ({ ...current, [feature.id]: Number(event.target.value) }))}
                    />
                    <output>{amount[feature.id] ?? 0}</output>
                  </label>
                )}
              </FeatureRow>
            ))}
          </>
        )}
        {tab === "video" && videoPage !== "basic" && <p className="clip-note">{videoPage === "cutout" ? "Cutout" : videoPage === "mask" ? "Mask" : "Enhance"} is part of the picture. It does not change the frame.</p>}
        {tab === "audio" && audioPage === "basic" && (
          <>
            <section className="clip-section">
              <header>
                <button type="button" className="check" aria-pressed={audioOn} aria-label={audioOn ? "Basic audio on" : "Basic audio off"} onClick={() => setAudioOn((value) => !value)} />
                <button type="button" className="section-name" aria-expanded={audioOpen} onClick={() => setAudioOpen((value) => !value)}>
                  <Icon name="chevron" size={14} />
                  Basic
                </button>
                <span className="key-tools">
                  <button type="button" aria-label="Reset audio" onClick={() => { setVolume(0); setFadeIn(0); setFadeOut(0) }}>
                    <Icon name="reset" size={14} />
                  </button>
                  <button type="button" aria-label="Previous keyframe" onClick={() => onNote("No keyframes on this clip.")}>
                    <Icon name="chevronLeft" size={14} />
                  </button>
                  <button type="button" aria-label="Add keyframe" onClick={() => onNote("The prototype does not store keyframes.")}>
                    <Icon name="keyframe" size={14} />
                  </button>
                  <button type="button" aria-label="Next keyframe" onClick={() => onNote("No keyframes on this clip.")}>
                    <Icon name="chevron" size={14} />
                  </button>
                </span>
              </header>
              {audioOpen && (
                <div className="section-body">
                  <label className="slider-row">
                    <span>Volume</span>
                    <input type="range" min={-60} max={12} step={0.1} value={volume} aria-label="Volume" onChange={(event) => setVolume(Number(event.target.value))} />
                    <output>{volume.toFixed(1)}dB</output>
                  </label>
                  <label className="slider-row">
                    <span>Fade in</span>
                    <input type="range" min={0} max={10} step={0.1} value={fadeIn} aria-label="Fade in" onChange={(event) => setFadeIn(Number(event.target.value))} />
                    <output>{fadeIn.toFixed(1)}s</output>
                  </label>
                  <label className="slider-row">
                    <span>Fade out</span>
                    <input type="range" min={0} max={10} step={0.1} value={fadeOut} aria-label="Fade out" onChange={(event) => setFadeOut(Number(event.target.value))} />
                    <output>{fadeOut.toFixed(1)}s</output>
                  </label>
                </div>
              )}
            </section>
            {audioFeatures.map((feature) => (
              <FeatureRow
                key={feature.id}
                label={feature.label}
                badge={feature.badge}
                open={openFeature === feature.id}
                enabled={enabled[feature.id] ?? false}
                onToggle={() => toggleFeature(feature.id)}
                onOpen={() => setOpenFeature((current) => (current === feature.id ? null : feature.id))}
              >
                <p className="clip-note">{feature.note ?? "This control is part of the picture."}</p>
              </FeatureRow>
            ))}
          </>
        )}
        {tab === "audio" && audioPage === "voice" && <p className="clip-note">Voice changer is part of the picture. It does not change the sound.</p>}
        {tab === "speed" && (
          <label className="slider-row speed-row">
            <span>Speed</span>
            <input type="range" min={0.1} max={4} step={0.1} value={speed} aria-label="Speed" onChange={(event) => setSpeed(Number(event.target.value))} />
            <output>{speed.toFixed(1)}×</output>
          </label>
        )}
        {tab === "style" && <p className="clip-note">AI stylize sits here. The prototype does not restyle the frame.</p>}
      </div>
      )}
    </aside>
  )
}

function AnimationTab({
  slot,
  query,
  pick,
  duration,
  onSlot,
  onQuery,
  onPick,
  onDuration,
}: {
  slot: AnimSlot
  query: string
  pick: string | null
  duration: number
  onSlot: (slot: AnimSlot) => void
  onQuery: (query: string) => void
  onPick: (id: string | null) => void
  onDuration: (duration: number) => void
}) {
  const needle = query.trim().toLowerCase()
  const items = animations[slot].filter((item) => item.label.toLowerCase().includes(needle))
  const showNone = needle === "" || "none".includes(needle)

  return (
    <div className="anim-panel">
      <div className="clip-pages" role="tablist" aria-label="Animation">
        {animSlots.map((item) => (
          <button key={item.id} type="button" role="tab" aria-selected={slot === item.id} data-active={slot === item.id ? "true" : "false"} onClick={() => onSlot(item.id)}>
            {item.label}
          </button>
        ))}
      </div>
      <label className="anim-search">
        <Icon name="search" size={14} />
        <input value={query} placeholder="Search animations" aria-label="Search animations" onChange={(event) => onQuery(event.target.value)} />
      </label>
      <div className="anim-grid">
        {showNone && (
          <button type="button" className="anim-card" aria-pressed={pick === null} data-picked={pick === null ? "true" : "false"} onClick={() => onPick(null)}>
            <span className="anim-thumb" data-look="none">
              <Icon name="unavailable" size={22} />
            </span>
            <span>None</span>
          </button>
        )}
        {items.map((item) => (
          <button key={item.id} type="button" className="anim-card" aria-pressed={pick === item.id} data-picked={pick === item.id ? "true" : "false"} onClick={() => onPick(item.id)}>
            <span className="anim-thumb" data-look={item.look} />
            <span>{item.label}</span>
          </button>
        ))}
        {!showNone && items.length === 0 && <p className="clip-note">No animations match.</p>}
      </div>
      <label className="anim-duration">
        <span>Duration</span>
        <output>{duration.toFixed(1)}s</output>
        <input
          type="range"
          min={0.1}
          max={10}
          step={0.1}
          value={duration}
          aria-label="Animation duration"
          disabled={pick === null}
          onChange={(event) => onDuration(Number(event.target.value))}
        />
      </label>
    </div>
  )
}

function FeatureRow({
  label,
  badge,
  apply,
  open,
  enabled,
  onToggle,
  onOpen,
  onApply,
  children,
}: {
  label: string
  badge?: string
  apply?: boolean
  open: boolean
  enabled: boolean
  onToggle: () => void
  onOpen: () => void
  onApply?: () => void
  children: ReactNode
}) {
  return (
    <section className="feature" data-open={open ? "true" : "false"}>
      <header>
        <button type="button" className="check" aria-pressed={enabled} aria-label={enabled ? `${label} on` : `${label} off`} onClick={onToggle} />
        <button type="button" className="section-name" aria-expanded={open} onClick={onOpen}>
          <span>{label}</span>
          {badge && <em>{badge}</em>}
          <Icon name="chevron" size={14} />
        </button>
        {apply && (
          <button type="button" className="apply" onClick={onApply}>
            Apply to all
          </button>
        )}
      </header>
      {open && <div className="section-body">{children}</div>}
    </section>
  )
}

function AdjustTab({
  page,
  on,
  open,
  lut,
  strength,
  skin,
  grades,
  onPage,
  onToggle,
  onOpen,
  onLut,
  onStrength,
  onSkin,
  onGrade,
  onResetLut,
  onResetGrade,
  onTint,
  onNote,
}: {
  page: AdjustPage
  on: Record<string, boolean>
  open: Record<string, boolean>
  lut: string
  strength: number
  skin: boolean
  grades: Record<string, number>
  onPage: (page: AdjustPage) => void
  onToggle: (id: string) => void
  onOpen: (id: string) => void
  onLut: (name: string) => void
  onStrength: (strength: number) => void
  onSkin: (skin: boolean) => void
  onGrade: (id: string, value: number) => void
  onResetLut: () => void
  onResetGrade: () => void
  onTint: (patch: Partial<Look>) => void
  onNote: (note: string) => void
}) {
  return (
    <div className="adjust-panel">
      <div className="clip-pages adjust-pages" role="tablist" aria-label="Adjust">
        {adjustPages.map((item) => (
          <button key={item.id} type="button" role="tab" aria-selected={page === item.id} data-active={page === item.id ? "true" : "false"} onClick={() => onPage(item.id)}>
            {item.label}
          </button>
        ))}
      </div>
      {page === "basic" ? (
        <>
          <div className="clip-body">
            <AdjustRow label="Auto adjust" enabled={on.auto ?? false} open={open.auto ?? false} onToggle={() => onToggle("auto")} onOpen={() => onOpen("auto")} onReset={() => onNote("Auto adjust is already clear.")} onNote={onNote}>
              <p className="clip-note">Auto adjust is part of the picture.</p>
            </AdjustRow>
            <AdjustRow label="Color match" enabled={on.match ?? false} open={open.match ?? false} onToggle={() => onToggle("match")} onOpen={() => onOpen("match")} onReset={() => onNote("Color match is already clear.")} onNote={onNote}>
              <p className="clip-note">Color match is part of the picture.</p>
            </AdjustRow>
            <AdjustRow label="Color correction" enabled={on.correct ?? false} open={open.correct ?? false} onToggle={() => onToggle("correct")} onOpen={() => onOpen("correct")} onReset={() => onNote("Color correction is already clear.")} onNote={onNote}>
              <p className="clip-note">Color correction is part of the picture.</p>
            </AdjustRow>
            <AdjustRow label="LUT" enabled={on.lut ?? false} open={open.lut ?? false} onToggle={() => onToggle("lut")} onOpen={() => onOpen("lut")} onReset={onResetLut} onNote={onNote}>
              <div className="lut-body" data-off={on.lut ? "false" : "true"}>
                <label className="lut-name">
                  Name
                  <select value={lut} aria-label="LUT name" onChange={(event) => onLut(event.target.value)}>
                    {lutNames.map((name) => (
                      <option key={name}>{name}</option>
                    ))}
                  </select>
                </label>
                <label className="grade-row">
                  <span>Strength</span>
                  <input type="range" min={0} max={100} value={strength} aria-label="LUT strength" onChange={(event) => onStrength(Number(event.target.value))} />
                  <output>{strength}</output>
                </label>
                <div className="skin-row">
                  <span>
                    Skin tone protection
                    <small>(Turn it on to make skin tone more natural)</small>
                  </span>
                  <button type="button" className="switch" aria-pressed={skin} aria-label={skin ? "Skin tone protection on" : "Skin tone protection off"} onClick={() => onSkin(!skin)}>
                    <span />
                  </button>
                </div>
              </div>
            </AdjustRow>
            <AdjustRow label="Adjust" enabled={on.grade ?? false} open={open.grade ?? false} onToggle={() => onToggle("grade")} onOpen={() => onOpen("grade")} onReset={onResetGrade} onNote={onNote}>
              <div className="lut-body" data-off={on.grade ? "false" : "true"}>
                {gradeGroups.map((group) => (
                  <div key={group.name} className="grade-group">
                    <p className="grade-name">{group.name}</p>
                    {group.sliders.map((slider) => (
                      <label key={slider.id} className="grade-row">
                        <span>{slider.label}</span>
                        <input
                          className="grade"
                          style={{ ["--grade" as string]: slider.grade }}
                          type="range"
                          min={slider.min}
                          max={slider.max}
                          value={grades[slider.id] ?? 0}
                          aria-label={slider.label}
                          onChange={(event) => onGrade(slider.id, Number(event.target.value))}
                        />
                        <output>{grades[slider.id] ?? 0}</output>
                      </label>
                    ))}
                  </div>
                ))}
              </div>
            </AdjustRow>
          </div>
          <footer className="adjust-foot">
            <button type="button" className="apply" onClick={() => onNote("The prototype does not change the other clips.")}>
              Apply to all
            </button>
            <button type="button" className="save-preset" onClick={() => onNote("The prototype does not save a preset.")}>
              Save as preset
            </button>
          </footer>
        </>
      ) : (
        <ColorTools page={page} onTint={onTint} />
      )}
    </div>
  )
}

function AdjustRow({
  label,
  enabled,
  open,
  onToggle,
  onOpen,
  onReset,
  onNote,
  children,
}: {
  label: string
  enabled: boolean
  open: boolean
  onToggle: () => void
  onOpen: () => void
  onReset: () => void
  onNote: (note: string) => void
  children: ReactNode
}) {
  return (
    <section className="feature">
      <header>
        <button type="button" className="check" aria-pressed={enabled} aria-label={enabled ? `${label} on` : `${label} off`} onClick={onToggle} />
        <button type="button" className="section-name" aria-expanded={open} onClick={onOpen}>
          <span>{label}</span>
          <Icon name="chevron" size={14} />
        </button>
        <span className="key-tools">
          <button type="button" aria-label={`Reset ${label}`} onClick={onReset}>
            <Icon name="reset" size={14} />
          </button>
          <button type="button" aria-label="Previous keyframe" onClick={() => onNote("No keyframes on this clip.")}>
            <Icon name="chevronLeft" size={14} />
          </button>
          <button type="button" aria-label="Add keyframe" onClick={() => onNote("The prototype does not store keyframes.")}>
            <Icon name="keyframe" size={14} />
          </button>
          <button type="button" aria-label="Next keyframe" onClick={() => onNote("No keyframes on this clip.")}>
            <Icon name="chevron" size={14} />
          </button>
        </span>
      </header>
      {open && <div className="section-body">{children}</div>}
    </section>
  )
}
