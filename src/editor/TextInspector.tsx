import { useState, type Dispatch, type ReactNode, type SetStateAction } from "react"
import { Icon } from "./icons"
import { useLook } from "./look"
import type { Clip } from "./model"

type TextTab = "text" | "animation" | "tracking" | "speech"
type TextPage = "basic" | "bubble" | "effects"
type AnimSlot = "in" | "out" | "loop"
type CaseMode = "upper" | "lower" | "title"
type VoiceTag = "Trending" | "New" | "Narration" | "Character"

const textTabs: { id: TextTab; label: string }[] = [
  { id: "text", label: "Text" },
  { id: "animation", label: "Animation" },
  { id: "tracking", label: "Tracking" },
  { id: "speech", label: "Text to speech" },
]

const textPages: { id: TextPage; label: string }[] = [
  { id: "basic", label: "Basic" },
  { id: "bubble", label: "Bubble" },
  { id: "effects", label: "Effects" },
]

const fonts = ["System", "Instrument Sans", "Newsreader"]

const presets: { id: string; label: string; color: string; plate: string }[] = [
  { id: "none", label: "None", color: "#fff", plate: "transparent" },
  { id: "plain", label: "Plain", color: "#fff", plate: "#3a3a3a" },
  { id: "yellow", label: "Yellow", color: "#ffe14a", plate: "#3a3a3a" },
  { id: "red", label: "Red", color: "#ff4d4d", plate: "#3a3a3a" },
  { id: "orange", label: "Orange", color: "#ff9a3c", plate: "#3a3a3a" },
  { id: "blue", label: "Blue", color: "#5aa7ff", plate: "#3a3a3a" },
  { id: "green", label: "Green", color: "#3ddc6a", plate: "#1c1c1c" },
  { id: "gray", label: "Gray", color: "#d0d0d0", plate: "#2a2a2a" },
  { id: "ink", label: "Ink", color: "#1a1a1a", plate: "#f3f3f3" },
  { id: "gold", label: "Gold", color: "#f2c14e", plate: "#2a2418" },
  { id: "violet", label: "Violet", color: "#fff", plate: "#7a3cff" },
  { id: "paper", label: "Paper", color: "#7a3cff", plate: "#fff" },
  { id: "night", label: "Night", color: "#fff", plate: "#111" },
  { id: "lime", label: "Lime", color: "#d6ff4a", plate: "#111" },
  { id: "rose", label: "Rose", color: "#ff6a9a", plate: "#2a1218" },
  { id: "cyan", label: "Cyan", color: "#3de0ff", plate: "#102028" },
]

const glows = [
  { id: "soft", label: "Soft glow" },
  { id: "halo", label: "Halo" },
]

const animSlots: { id: AnimSlot; label: string }[] = [
  { id: "in", label: "In" },
  { id: "out", label: "Out" },
  { id: "loop", label: "Loop" },
]

const motions: Record<AnimSlot, { id: string; label: string; look: string }[]> = {
  in: [
    { id: "fade-in", label: "Fade", look: "fade" },
    { id: "rise-in", label: "Rise", look: "rise" },
    { id: "drop-in", label: "Drop", look: "down" },
    { id: "slide-in", label: "Slide", look: "left" },
    { id: "pop-in", label: "Pop", look: "pop" },
    { id: "type-in", label: "Type on", look: "wipe" },
    { id: "blur-in", label: "Blur", look: "blur" },
    { id: "spin-in", label: "Spin", look: "spin" },
    { id: "bounce-in", label: "Bounce", look: "bounce" },
    { id: "unfold", label: "Unfold", look: "zoom" },
  ],
  out: [
    { id: "fade-out", label: "Fade", look: "fade" },
    { id: "rise-out", label: "Rise", look: "up" },
    { id: "drop-out", label: "Drop", look: "down" },
    { id: "slide-out", label: "Slide", look: "right" },
    { id: "shrink-out", label: "Shrink", look: "zoom" },
    { id: "type-out", label: "Type off", look: "wipe" },
    { id: "blur-out", label: "Blur", look: "blur" },
    { id: "spin-out", label: "Spin", look: "spin" },
  ],
  loop: [
    { id: "pulse", label: "Pulse", look: "pulse" },
    { id: "flicker", label: "Flicker", look: "flicker" },
    { id: "float", label: "Float", look: "float" },
    { id: "shake", label: "Shake", look: "shake" },
    { id: "swing", label: "Swing", look: "spin" },
    { id: "breathe", label: "Breathe", look: "pulse" },
  ],
}

const voices: { id: string; name: string; tag: VoiceTag }[] = [
  { id: "warm", name: "Warm", tag: "Trending" },
  { id: "bright", name: "Bright", tag: "Trending" },
  { id: "even", name: "Even", tag: "Trending" },
  { id: "clear", name: "Clear", tag: "New" },
  { id: "light", name: "Light", tag: "New" },
  { id: "low", name: "Low", tag: "Narration" },
  { id: "soft", name: "Soft", tag: "Narration" },
  { id: "steady", name: "Steady", tag: "Narration" },
  { id: "quick", name: "Quick", tag: "Character" },
  { id: "dry", name: "Dry", tag: "Character" },
]

const alignButtons = [
  ["textLeft", "Align left"],
  ["textCenter", "Align center"],
  ["textRight", "Align right"],
  ["justifyLeft", "Justify left"],
  ["justifyCenter", "Justify center"],
  ["justifyRight", "Justify right"],
] as const

export function TextInspector({
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
  const [tab, setTab] = useState<TextTab>("text")
  const [page, setPage] = useState<TextPage>("basic")
  const [copy, setCopy] = useState(clip.name)
  const [font, setFont] = useState(fonts[0])
  const [size, setSize] = useState(15)
  const [bold, setBold] = useState(false)
  const [italic, setItalic] = useState(false)
  const [underline, setUnderline] = useState(false)
  const [letterCase, setLetterCase] = useState<CaseMode | null>(null)
  const [color, setColor] = useState("#ffffff")
  const [character, setCharacter] = useState(0)
  const [line, setLine] = useState(0)
  const [align, setAlign] = useState<(typeof alignButtons)[number][0]>("textLeft")
  const [preset, setPreset] = useState("none")
  const [positionOpen, setPositionOpen] = useState(true)
  const [scale, setScale] = useState(100)
  const [uniform, setUniform] = useState(true)
  const [x, setX] = useState(0)
  const [y, setY] = useState(0)
  const [rotate, setRotate] = useState(0)
  const [openFeature, setOpenFeature] = useState<string | null>("blend")
  const [enabled, setEnabled] = useState<Record<string, boolean>>({ blend: true })
  const [opacity, setOpacity] = useState(100)
  const [stroke, setStroke] = useState(40)
  const [strokeColor, setStrokeColor] = useState("#ffffff")
  const [background, setBackground] = useState("#000000")
  const [backgroundOpacity, setBackgroundOpacity] = useState(100)
  const [glow, setGlow] = useState("soft")
  const [glowColor, setGlowColor] = useState("#ffffff")
  const [intensity, setIntensity] = useState(30)
  const [range, setRange] = useState(20)
  const [glowY, setGlowY] = useState(0)
  const [glowX, setGlowX] = useState(0)
  const [shadowColor, setShadowColor] = useState("#000000")
  const [shadowOpacity, setShadowOpacity] = useState(90)
  const [blur, setBlur] = useState(15)
  const [distance, setDistance] = useState(5)
  const [angle, setAngle] = useState(-45)
  const { setLook } = useLook()
  const [bubble, setBubble] = useState("none")
  const [textEffect, setTextEffect] = useState("none")
  const [trackMode, setTrackMode] = useState("Face")

  const applyCase = (mode: CaseMode) => {
    setLetterCase(mode)
    setCopy((current) => {
      if (mode === "upper") return current.toUpperCase()
      if (mode === "lower") return current.toLowerCase()
      return current.replace(/\S+/g, (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    })
  }

  const pickPreset = (id: string) => {
    setPreset(id)
    const next = presets.find((item) => item.id === id)
    if (next && id !== "none") setColor(next.color)
  }

  return (
    <aside className="inspector clip-inspector" style={{ width }} aria-label={`${clip.name} inspector`}>
      <div className="clip-tabs" role="tablist" aria-label="Text inspector">
        {textTabs.map((item) => (
          <button key={item.id} type="button" role="tab" aria-selected={tab === item.id} data-active={tab === item.id ? "true" : "false"} onClick={() => setTab(item.id)}>
            {item.label}
          </button>
        ))}
      </div>
      {selectedCount > 1 && <p className="clip-count">{selectedCount} clips selected.</p>}
      {tab === "text" && (
        <div className="clip-pages" role="tablist" aria-label="Text">
          {textPages.map((item) => (
            <button key={item.id} type="button" role="tab" aria-selected={page === item.id} data-active={page === item.id ? "true" : "false"} onClick={() => setPage(item.id)}>
              {item.label}
            </button>
          ))}
        </div>
      )}
      {tab === "animation" && <TextAnimation onNote={onNote} />}
      {tab === "tracking" && (
        <div className="clip-body">
          <div className="voice-tags" role="toolbar" aria-label="Tracking target">
            {["Face", "Body", "Hand", "Custom"].map((item) => (
              <button key={item} type="button" aria-pressed={trackMode === item} onClick={() => setTrackMode(item)}>
                {item}
              </button>
            ))}
          </div>
          <p className="clip-note">The line stays with the {trackMode.toLowerCase()} target on the mock frame. The prototype does not follow a subject.</p>
          <button type="button" className="save-preset" onClick={() => onNote("The prototype does not follow a subject.")}>Start tracking</button>
        </div>
      )}
      {tab === "speech" && <SpeechPanel copy={copy} onNote={onNote} />}
      {tab === "text" && page === "bubble" && (
        <div className="clip-body">
          <div className="preset-grid">
            {[
              ["none", "None"],
              ["round", "Round"],
              ["box", "Box"],
              ["thought", "Thought"],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                className="preset-tile"
                aria-label={label}
                aria-pressed={bubble === id}
                data-picked={bubble === id ? "true" : "false"}
                onClick={() => {
                  setBubble(id)
                  setLook(clip.id, { bubble: id })
                }}
              >
                {id === "none" ? <Icon name="unavailable" size={16} /> : "Aa"}
              </button>
            ))}
          </div>
        </div>
      )}
      {tab === "text" && page === "effects" && (
        <div className="clip-body">
          <div className="preset-grid">
            {[
              ["none", "None"],
              ["shadow", "Shadow"],
              ["glow", "Glow"],
              ["outline", "Outline"],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                className="preset-tile"
                aria-label={label}
                aria-pressed={textEffect === id}
                data-picked={textEffect === id ? "true" : "false"}
                onClick={() => {
                  setTextEffect(id)
                  setLook(clip.id, { textEffect: id })
                }}
              >
                {label === "None" ? <Icon name="unavailable" size={16} /> : "Aa"}
              </button>
            ))}
          </div>
        </div>
      )}
      {tab === "text" && page === "basic" && (
        <>
          <div className="clip-body">
            <label className="text-copy">
              <textarea
                aria-label="Text"
                value={copy}
                style={{
                  fontFamily: font === "System" ? "inherit" : font,
                  fontSize: size,
                  fontWeight: bold ? 700 : 500,
                  fontStyle: italic ? "italic" : "normal",
                  textDecoration: underline ? "underline" : "none",
                  color,
                }}
                onChange={(event) => setCopy(event.target.value)}
              />
            </label>
            <label className="field-row">
              <span>Font</span>
              <select aria-label="Font" value={font} onChange={(event) => setFont(event.target.value)}>
                {fonts.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>
            </label>
            <label className="slider-row">
              <span>Font size</span>
              <input type="range" min={6} max={96} value={size} aria-label="Font size" onChange={(event) => setSize(Number(event.target.value))} />
              <input className="mini-num" type="number" min={6} max={96} value={size} aria-label="Font size value" onChange={(event) => setSize(Number(event.target.value))} />
            </label>
            <div className="field-row">
              <span>Style</span>
              <div className="style-bar" role="toolbar" aria-label="Style">
                <button type="button" aria-label="Bold" aria-pressed={bold} onClick={() => setBold((value) => !value)}>
                  <Icon name="bold" size={14} />
                </button>
                <button type="button" aria-label="Underline" aria-pressed={underline} onClick={() => setUnderline((value) => !value)}>
                  <Icon name="underline" size={14} />
                </button>
                <button type="button" aria-label="Italic" aria-pressed={italic} onClick={() => setItalic((value) => !value)}>
                  <Icon name="italic" size={14} />
                </button>
              </div>
            </div>
            <div className="field-row">
              <span>Case</span>
              <div className="case-bar" role="toolbar" aria-label="Case">
                {(
                  [
                    ["upper", "TT"],
                    ["lower", "tt"],
                    ["title", "Tt"],
                  ] as const
                ).map(([mode, label]) => (
                  <button key={mode} type="button" aria-label={mode === "upper" ? "Upper case" : mode === "lower" ? "Lower case" : "Title case"} aria-pressed={letterCase === mode} onClick={() => applyCase(mode)}>
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <div className="field-row">
              <span>Color</span>
              <label className="swatch">
                <input type="color" value={color} aria-label="Color" onChange={(event) => { setColor(event.target.value); setPreset("none") }} />
              </label>
            </div>
            <div className="step-row">
              <span>Character</span>
              <input type="number" value={character} aria-label="Character spacing" onChange={(event) => setCharacter(Number(event.target.value))} />
              <span>Line</span>
              <input type="number" value={line} aria-label="Line spacing" onChange={(event) => setLine(Number(event.target.value))} />
            </div>
            <div className="field-row">
              <span>Alignment</span>
              <div className="align-bar text-align" role="toolbar" aria-label="Alignment">
                {alignButtons.map(([name, label]) => (
                  <button key={name} type="button" aria-label={label} aria-pressed={align === name} onClick={() => setAlign(name)}>
                    <Icon name={name} size={16} />
                  </button>
                ))}
              </div>
            </div>
            <div className="preset-block">
              <p>Preset style</p>
              <div className="preset-grid">
                {presets.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className="preset-tile"
                    aria-label={item.label}
                    aria-pressed={preset === item.id}
                    data-picked={preset === item.id ? "true" : "false"}
                    style={item.id === "none" ? undefined : { color: item.color, background: item.plate }}
                    onClick={() => pickPreset(item.id)}
                  >
                    {item.id === "none" ? <Icon name="unavailable" size={16} /> : "Aa"}
                  </button>
                ))}
              </div>
            </div>
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
                    <span>Plane rotation</span>
                    <label>
                      <input type="number" value={rotate} step={0.1} aria-label="Plane rotation" onChange={(event) => setRotate(Number(event.target.value))} />
                      <span>°</span>
                    </label>
                  </div>
                </div>
              )}
            </section>
            <FeatureRow label="Blend" open={openFeature === "blend"} enabled={enabled.blend ?? false} onToggle={() => toggle(setEnabled, setOpenFeature, "blend")} onOpen={() => setOpenFeature((current) => (current === "blend" ? null : "blend"))}>
              <label className="slider-row">
                <span>Opacity</span>
                <input type="range" min={0} max={100} value={opacity} aria-label="Opacity" onChange={(event) => setOpacity(Number(event.target.value))} />
                <output>{opacity}%</output>
              </label>
            </FeatureRow>
            <FeatureRow label="Stroke" open={openFeature === "stroke"} enabled={enabled.stroke ?? false} onToggle={() => toggle(setEnabled, setOpenFeature, "stroke")} onOpen={() => setOpenFeature((current) => (current === "stroke" ? null : "stroke"))}>
              <ColorField label="Color" value={strokeColor} onChange={setStrokeColor} />
              <label className="slider-row">
                <span>Thickness</span>
                <input type="range" min={0} max={100} value={stroke} aria-label="Thickness" onChange={(event) => setStroke(Number(event.target.value))} />
                <output>{stroke}</output>
              </label>
            </FeatureRow>
            <FeatureRow label="Background" open={openFeature === "background"} enabled={enabled.background ?? false} onToggle={() => toggle(setEnabled, setOpenFeature, "background")} onOpen={() => setOpenFeature((current) => (current === "background" ? null : "background"))}>
              <ColorField label="Color" value={background} onChange={setBackground} />
              <label className="slider-row">
                <span>Opacity</span>
                <input type="range" min={0} max={100} value={backgroundOpacity} aria-label="Background opacity" onChange={(event) => setBackgroundOpacity(Number(event.target.value))} />
                <output>{backgroundOpacity}%</output>
              </label>
            </FeatureRow>
            <FeatureRow label="Glow" open={openFeature === "glow"} enabled={enabled.glow ?? false} onToggle={() => toggle(setEnabled, setOpenFeature, "glow")} onOpen={() => setOpenFeature((current) => (current === "glow" ? null : "glow"))}>
              <div className="glow-picks">
                {glows.map((item) => (
                  <button key={item.id} type="button" className="glow-tile" aria-label={item.label} aria-pressed={glow === item.id} data-picked={glow === item.id ? "true" : "false"} data-glow={item.id} onClick={() => setGlow(item.id)}>
                    Aa
                  </button>
                ))}
              </div>
              <ColorField label="Color" value={glowColor} onChange={setGlowColor} />
              <label className="slider-row">
                <span>Intensity</span>
                <input type="range" min={0} max={100} value={intensity} aria-label="Intensity" onChange={(event) => setIntensity(Number(event.target.value))} />
                <output>{intensity}</output>
              </label>
              <label className="slider-row">
                <span>Range</span>
                <input type="range" min={0} max={100} value={range} aria-label="Range" onChange={(event) => setRange(Number(event.target.value))} />
                <output>{range}</output>
              </label>
              <label className="slider-row">
                <span>Y-offset</span>
                <input type="range" min={-180} max={180} value={glowY} aria-label="Y-offset" onChange={(event) => setGlowY(Number(event.target.value))} />
                <output>{glowY}°</output>
              </label>
              <label className="slider-row">
                <span>X-offset</span>
                <input type="range" min={-180} max={180} value={glowX} aria-label="X-offset" onChange={(event) => setGlowX(Number(event.target.value))} />
                <output>{glowX}°</output>
              </label>
            </FeatureRow>
            <FeatureRow label="Shadow" open={openFeature === "shadow"} enabled={enabled.shadow ?? false} onToggle={() => toggle(setEnabled, setOpenFeature, "shadow")} onOpen={() => setOpenFeature((current) => (current === "shadow" ? null : "shadow"))}>
              <ColorField label="Color" value={shadowColor} onChange={setShadowColor} />
              <label className="slider-row">
                <span>Opacity</span>
                <input type="range" min={0} max={100} value={shadowOpacity} aria-label="Shadow opacity" onChange={(event) => setShadowOpacity(Number(event.target.value))} />
                <output>{shadowOpacity}%</output>
              </label>
              <label className="slider-row">
                <span>Bluriness</span>
                <input type="range" min={0} max={100} value={blur} aria-label="Bluriness" onChange={(event) => setBlur(Number(event.target.value))} />
                <output>{blur}%</output>
              </label>
              <label className="slider-row">
                <span>Distance</span>
                <input type="range" min={0} max={100} value={distance} aria-label="Distance" onChange={(event) => setDistance(Number(event.target.value))} />
                <output>{distance}</output>
              </label>
              <label className="slider-row">
                <span>Angle</span>
                <input type="range" min={-180} max={180} value={angle} aria-label="Angle" onChange={(event) => setAngle(Number(event.target.value))} />
                <output>{angle}°</output>
              </label>
            </FeatureRow>
          </div>
          <footer className="adjust-foot">
            <button type="button" className="save-preset" onClick={() => onNote("The prototype does not save a text preset.")}>
              Save as preset
            </button>
          </footer>
        </>
      )}
    </aside>
  )
}

function toggle(
  setEnabled: Dispatch<SetStateAction<Record<string, boolean>>>,
  setOpen: Dispatch<SetStateAction<string | null>>,
  id: string,
) {
  setEnabled((current) => ({ ...current, [id]: !current[id] }))
  setOpen(id)
}

function ColorField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <div className="field-row">
      <span>{label}</span>
      <label className="swatch">
        <input type="color" value={value} aria-label={label} onChange={(event) => onChange(event.target.value)} />
      </label>
    </div>
  )
}

function FeatureRow({
  label,
  open,
  enabled,
  onToggle,
  onOpen,
  children,
}: {
  label: string
  open: boolean
  enabled: boolean
  onToggle: () => void
  onOpen: () => void
  children: ReactNode
}) {
  return (
    <section className="feature" data-open={open ? "true" : "false"}>
      <header>
        <button type="button" className="check" aria-pressed={enabled} aria-label={enabled ? `${label} on` : `${label} off`} onClick={onToggle} />
        <button type="button" className="section-name" aria-expanded={open} onClick={onOpen}>
          <span>{label}</span>
          <Icon name="chevron" size={14} />
        </button>
      </header>
      {open && (
        <div className="section-body" data-off={enabled ? "false" : "true"}>
          {children}
        </div>
      )}
    </section>
  )
}

function TextAnimation({ onNote }: { onNote: (note: string) => void }) {
  const [slot, setSlot] = useState<AnimSlot>("in")
  const [query, setQuery] = useState("")
  const [pick, setPick] = useState<Record<AnimSlot, string | null>>({ in: null, out: null, loop: null })
  const needle = query.trim().toLowerCase()
  const items = motions[slot].filter((item) => item.label.toLowerCase().includes(needle))
  const showNone = needle === "" || "none".includes(needle)
  const chosen = pick[slot]

  return (
    <div className="anim-panel">
      <div className="clip-pages" role="tablist" aria-label="Animation">
        {animSlots.map((item) => (
          <button key={item.id} type="button" role="tab" aria-selected={slot === item.id} data-active={slot === item.id ? "true" : "false"} onClick={() => setSlot(item.id)}>
            {item.label}
          </button>
        ))}
      </div>
      <label className="anim-search">
        <Icon name="search" size={14} />
        <input value={query} placeholder="Search animations" aria-label="Search animations" onChange={(event) => setQuery(event.target.value)} />
      </label>
      <div className="anim-grid">
        {showNone && (
          <button type="button" className="anim-card" aria-pressed={chosen === null} data-picked={chosen === null ? "true" : "false"} onClick={() => { setPick((current) => ({ ...current, [slot]: null })); onNote("No text animation.") }}>
            <span className="anim-thumb" data-look="none">
              <Icon name="unavailable" size={22} />
            </span>
            <span>None</span>
          </button>
        )}
        {items.map((item) => (
          <button key={item.id} type="button" className="anim-card" aria-pressed={chosen === item.id} data-picked={chosen === item.id ? "true" : "false"} onClick={() => setPick((current) => ({ ...current, [slot]: item.id }))}>
            <span className="anim-thumb" data-look={item.look} />
            <span>{item.label}</span>
          </button>
        ))}
        {!showNone && items.length === 0 && <p className="clip-note">No animations match.</p>}
      </div>
    </div>
  )
}

function SpeechPanel({ copy, onNote }: { copy: string; onNote: (note: string) => void }) {
  const [query, setQuery] = useState("")
  const [tag, setTag] = useState<VoiceTag>("Trending")
  const [voice, setVoice] = useState("warm")
  const [stars, setStars] = useState<Record<string, boolean>>({})
  const needle = query.trim().toLowerCase()
  const items = voices.filter((item) => item.tag === tag && item.name.toLowerCase().includes(needle))

  return (
    <div className="speech-panel">
      <label className="anim-search">
        <Icon name="search" size={14} />
        <input value={query} placeholder="Search voices" aria-label="Search voices" onChange={(event) => setQuery(event.target.value)} />
      </label>
      <div className="voice-tags" role="toolbar" aria-label="Voice groups">
        {(["Trending", "New", "Narration", "Character"] as const).map((item) => (
          <button key={item} type="button" aria-pressed={tag === item} onClick={() => setTag(item)}>
            {item}
          </button>
        ))}
      </div>
      <div className="voice-grid">
        {items.map((item) => (
          <button key={item.id} type="button" className="voice-card" aria-pressed={voice === item.id} data-picked={voice === item.id ? "true" : "false"} onClick={() => setVoice(item.id)}>
            <span>{item.name}</span>
            <span
              className="voice-star"
              role="checkbox"
              aria-checked={stars[item.id] ?? false}
              aria-label={stars[item.id] ? `Unstar ${item.name}` : `Star ${item.name}`}
              onClick={(event) => {
                event.stopPropagation()
                setStars((current) => ({ ...current, [item.id]: !current[item.id] }))
              }}
            >
              <Icon name="star" size={12} />
            </span>
          </button>
        ))}
        {items.length === 0 && <p className="clip-note">No voices match.</p>}
      </div>
      <footer className="adjust-foot">
        <button type="button" className="save-preset" onClick={() => onNote(copy.trim() ? "The prototype does not generate speech." : "Write the line first.")}>
          Generate speech
        </button>
      </footer>
    </div>
  )
}
