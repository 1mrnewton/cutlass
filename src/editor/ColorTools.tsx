import { useState } from "react"
import type { Look } from "./look"

const hues = ["Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Magenta"]

export function ColorTools({
  page,
  onTint,
}: {
  page: "hsl" | "curves" | "wheel" | "mask"
  onTint: (patch: Partial<Look>) => void
}) {
  if (page === "hsl") return <Hsl onTint={onTint} />
  if (page === "curves") return <Curves onTint={onTint} />
  if (page === "wheel") return <Wheels onTint={onTint} />
  return <Mask />
}

function Hsl({ onTint }: { onTint: (patch: Partial<Look>) => void }) {
  const [values, setValues] = useState<Record<string, number>>({})
  const set = (id: string, value: number) => {
    const next = { ...values, [id]: value }
    setValues(next)
    const sat = hues.reduce((sum, name) => sum + (next[`${name}-s`] ?? 0), 0) / hues.length
    const bright = hues.reduce((sum, name) => sum + (next[`${name}-l`] ?? 0), 0) / hues.length
    onTint({ sat, bright })
  }
  return (
    <div className="clip-body">
      {hues.map((name) => (
        <div key={name} className="hsl-row">
          <span>{name}</span>
          {(["h", "s", "l"] as const).map((channel) => (
            <label key={channel}>
              {channel.toUpperCase()}
              <input
                type="range"
                min={-100}
                max={100}
                value={values[`${name}-${channel}`] ?? 0}
                aria-label={`${name} ${channel}`}
                onChange={(event) => set(`${name}-${channel}`, Number(event.target.value))}
              />
            </label>
          ))}
        </div>
      ))}
    </div>
  )
}

function Curves({ onTint }: { onTint: (patch: Partial<Look>) => void }) {
  const [shadow, setShadow] = useState(0)
  const [mid, setMid] = useState(0)
  const [high, setHigh] = useState(0)
  const paint = (next: { shadow: number; mid: number; high: number }) => {
    onTint({ bright: (next.shadow + next.mid + next.high) / 3 })
  }
  const y = (value: number) => 80 - ((value + 100) / 200) * 70
  return (
    <div className="clip-body">
      <svg className="curve" viewBox="0 0 160 90" aria-hidden="true">
        <line x1="8" y1="82" x2="152" y2="8" />
        <path d={`M8 ${y(shadow)} Q 80 ${y(mid)} 152 ${y(high)}`} />
      </svg>
      <Slider label="Shadows" value={shadow} onChange={(value) => { setShadow(value); paint({ shadow: value, mid, high }) }} />
      <Slider label="Midtones" value={mid} onChange={(value) => { setMid(value); paint({ shadow, mid: value, high }) }} />
      <Slider label="Highlights" value={high} onChange={(value) => { setHigh(value); paint({ shadow, mid, high: value }) }} />
    </div>
  )
}

function Wheels({ onTint }: { onTint: (patch: Partial<Look>) => void }) {
  const [wheels, setWheels] = useState({ lift: 0, gamma: 0, gain: 0 })
  const set = (id: "lift" | "gamma" | "gain", value: number) => {
    const next = { ...wheels, [id]: value }
    setWheels(next)
    onTint({ temp: (next.lift + next.gain) / 2, bright: next.gamma })
  }
  return (
    <div className="clip-body wheel-row">
      {(
        [
          ["lift", "Lift"],
          ["gamma", "Gamma"],
          ["gain", "Gain"],
        ] as const
      ).map(([id, label]) => (
        <label key={id} className="wheel">
          <span className="wheel-disc" style={{ ["--turn" as string]: `${wheels[id]}deg` }} />
          <span>{label}</span>
          <input type="range" min={-180} max={180} value={wheels[id]} aria-label={label} onChange={(event) => set(id, Number(event.target.value))} />
        </label>
      ))}
    </div>
  )
}

function Mask() {
  const [shape, setShape] = useState("Circle")
  const [feather, setFeather] = useState(20)
  const [size, setSize] = useState(50)
  const [invert, setInvert] = useState(false)
  return (
    <div className="clip-body">
      <label className="field-row">
        <span>Shape</span>
        <select aria-label="Mask shape" value={shape} onChange={(event) => setShape(event.target.value)}>
          {["None", "Circle", "Rectangle"].map((item) => (
            <option key={item}>{item}</option>
          ))}
        </select>
      </label>
      <Slider label="Size" value={size} min={0} max={100} onChange={setSize} />
      <Slider label="Feather" value={feather} min={0} max={100} onChange={setFeather} />
      <div className="switch-row">
        <span>Invert</span>
        <button type="button" className="switch" aria-pressed={invert} aria-label={invert ? "Invert mask on" : "Invert mask off"} onClick={() => setInvert((value) => !value)}>
          <span />
        </button>
      </div>
      <p className="clip-note">The mask is drawn on the mock frame. It does not cut the picture.</p>
    </div>
  )
}

function Slider({
  label,
  value,
  min = -100,
  max = 100,
  onChange,
}: {
  label: string
  value: number
  min?: number
  max?: number
  onChange: (value: number) => void
}) {
  return (
    <label className="slider-row">
      <span>{label}</span>
      <input type="range" min={min} max={max} value={value} aria-label={label} onChange={(event) => onChange(Number(event.target.value))} />
      <output>{value}</output>
    </label>
  )
}
