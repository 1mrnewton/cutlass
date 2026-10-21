import { useState } from "react"
import { ColorTools } from "./ColorTools"
import { Icon } from "./icons"
import { useLook } from "./look"
import type { Clip } from "./model"

export function AudioInspector({ clip, width, onNote }: { clip: Clip; width: number; onNote: (note: string) => void }) {
  const { looks, setLook } = useLook()
  const volume = looks[clip.id]?.volume ?? 0
  const [fadeIn, setFadeIn] = useState(0)
  const [fadeOut, setFadeOut] = useState(0)
  const [denoise, setDenoise] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [tab, setTab] = useState<"audio" | "speed">("audio")
  return (
    <aside className="inspector clip-inspector" style={{ width }} aria-label={`${clip.name} inspector`}>
      <div className="clip-tabs" role="tablist" aria-label="Audio inspector">
        <button type="button" role="tab" aria-selected={tab === "audio"} data-active={tab === "audio" ? "true" : "false"} onClick={() => setTab("audio")}>Audio</button>
        <button type="button" role="tab" aria-selected={tab === "speed"} data-active={tab === "speed" ? "true" : "false"} onClick={() => setTab("speed")}>Speed</button>
      </div>
      <div className="clip-body">
        {tab === "speed" ? (
          <>
            <label className="slider-row">
              <span>Speed</span>
              <input type="range" min={0.1} max={4} step={0.1} value={speed} aria-label="Speed" onChange={(event) => setSpeed(Number(event.target.value))} />
              <output>{speed.toFixed(1)}×</output>
            </label>
            <p className="clip-note">Speed stays at 1× on the timeline. The prototype does not hear the clip.</p>
          </>
        ) : (
        <>
        <label className="slider-row">
          <span>Volume</span>
          <input type="range" min={-60} max={12} step={0.1} value={volume} aria-label="Volume" onChange={(event) => setLook(clip.id, { volume: Number(event.target.value) })} />
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
        <label className="slider-row">
          <span>Denoise</span>
          <input type="range" min={0} max={100} value={denoise} aria-label="Denoise" onChange={(event) => setDenoise(Number(event.target.value))} />
          <output>{denoise}</output>
        </label>
        <p className="clip-note">Volume moves the meter on the frame. The prototype does not play the file.</p>
        <button type="button" className="text-btn" onClick={() => onNote("The prototype does not change the voice.")}>Change voice</button>
        </>
        )}
      </div>
    </aside>
  )
}

export function StickerInspector({ clip, width }: { clip: Clip; width: number }) {
  const { looks, setLook } = useLook()
  const opacity = looks[clip.id]?.opacity ?? 100
  const [slot, setSlot] = useState<"in" | "out" | "loop">("in")
  const [tab, setTab] = useState<"sticker" | "animation">("sticker")
  return (
    <aside className="inspector clip-inspector" style={{ width }} aria-label={`${clip.name} inspector`}>
      <div className="clip-tabs" role="tablist">
        <button type="button" role="tab" aria-selected={tab === "sticker"} data-active={tab === "sticker" ? "true" : "false"} onClick={() => setTab("sticker")}>Sticker</button>
        <button type="button" role="tab" aria-selected={tab === "animation"} data-active={tab === "animation" ? "true" : "false"} onClick={() => setTab("animation")}>Animation</button>
      </div>
      {tab === "sticker" ? (
      <div className="clip-body">
        <label className="slider-row">
          <span>Opacity</span>
          <input type="range" min={0} max={100} value={opacity} aria-label="Opacity" onChange={(event) => setLook(clip.id, { opacity: Number(event.target.value) })} />
          <output>{opacity}%</output>
        </label>
      </div>
      ) : (
      <div className="clip-body">
        <div className="clip-pages" role="tablist" aria-label="Sticker animation">
          {(["in", "out", "loop"] as const).map((item) => (
            <button key={item} type="button" role="tab" aria-selected={slot === item} data-active={slot === item ? "true" : "false"} onClick={() => setSlot(item)}>
              {item === "in" ? "In" : item === "out" ? "Out" : "Loop"}
            </button>
          ))}
        </div>
        <div className="anim-grid">
          {["None", "Pop", "Spin", "Fade"].map((item) => (
            <button key={item} type="button" className="anim-card">
              <span className="anim-thumb" data-look={item === "None" ? "none" : item === "Pop" ? "pop" : item === "Spin" ? "spin" : "fade"}>
                {item === "None" && <Icon name="unavailable" size={18} />}
              </span>
              <span>{item}</span>
            </button>
          ))}
        </div>
      </div>
      )}
    </aside>
  )
}

export function EffectInspector({ clip, width }: { clip: Clip; width: number }) {
  const { looks, setLook } = useLook()
  const amount = looks[clip.id]?.amount ?? 70
  return (
    <aside className="inspector clip-inspector" style={{ width }} aria-label={`${clip.name} inspector`}>
      <div className="clip-tabs"><button type="button" data-active="true">Effect</button></div>
      <div className="clip-body">
        <p className="clip-note">{clip.name}</p>
        <label className="slider-row">
          <span>Amount</span>
          <input type="range" min={0} max={100} value={amount} aria-label="Amount" onChange={(event) => setLook(clip.id, { amount: Number(event.target.value) })} />
          <output>{amount}</output>
        </label>
      </div>
    </aside>
  )
}

export function FilterInspector({ clip, width }: { clip: Clip; width: number }) {
  const { looks, setLook } = useLook()
  const amount = looks[clip.id]?.amount ?? 60
  const [temp, setTemp] = useState(looks[clip.id]?.temp ?? (clip.name === "Cool" ? -30 : 20))
  return (
    <aside className="inspector clip-inspector" style={{ width }} aria-label={`${clip.name} inspector`}>
      <div className="clip-tabs"><button type="button" data-active="true">Filter</button></div>
      <div className="clip-body">
        <p className="clip-note">{clip.name}</p>
        <label className="slider-row">
          <span>Amount</span>
          <input
            type="range"
            min={0}
            max={100}
            value={amount}
            aria-label="Amount"
            onChange={(event) => setLook(clip.id, { amount: Number(event.target.value), temp })}
          />
          <output>{amount}</output>
        </label>
        <label className="slider-row">
          <span>Temp</span>
          <input
            type="range"
            min={-100}
            max={100}
            value={temp}
            aria-label="Temp"
            onChange={(event) => {
              const value = Number(event.target.value)
              setTemp(value)
              setLook(clip.id, { temp: value, amount })
            }}
          />
          <output>{temp}</output>
        </label>
      </div>
    </aside>
  )
}

export function AdjustClipInspector({ clip, width }: { clip: Clip; width: number }) {
  const { setLook } = useLook()
  const [page, setPage] = useState<"basic" | "hsl" | "curves" | "wheel" | "mask">("basic")
  const [temp, setTemp] = useState(clip.name === "Temperature" ? 35 : 0)
  const [bright, setBright] = useState(clip.name === "Exposure" ? 20 : 0)
  const [sat, setSat] = useState(clip.name === "Contrast" ? 25 : 0)
  const push = (patch: { temp?: number; bright?: number; sat?: number }) => setLook(clip.id, patch)
  return (
    <aside className="inspector clip-inspector" style={{ width }} aria-label={`${clip.name} inspector`}>
      <div className="clip-tabs"><button type="button" data-active="true">Adjust</button></div>
      <div className="clip-pages" role="tablist" aria-label="Adjust">
        {(
          [
            ["basic", "Basic"],
            ["hsl", "HSL"],
            ["curves", "Curves"],
            ["wheel", "Color wheel"],
            ["mask", "Mask"],
          ] as const
        ).map(([id, label]) => (
          <button key={id} type="button" role="tab" aria-selected={page === id} data-active={page === id ? "true" : "false"} onClick={() => setPage(id)}>
            {label}
          </button>
        ))}
      </div>
      {page === "basic" ? (
        <div className="clip-body">
          <label className="slider-row">
            <span>Temp</span>
            <input type="range" min={-100} max={100} value={temp} aria-label="Temp" onChange={(event) => { const value = Number(event.target.value); setTemp(value); push({ temp: value, bright, sat }) }} />
            <output>{temp}</output>
          </label>
          <label className="slider-row">
            <span>Exposure</span>
            <input type="range" min={-100} max={100} value={bright} aria-label="Exposure" onChange={(event) => { const value = Number(event.target.value); setBright(value); push({ temp, bright: value, sat }) }} />
            <output>{bright}</output>
          </label>
          <label className="slider-row">
            <span>Contrast</span>
            <input type="range" min={-100} max={100} value={sat} aria-label="Contrast" onChange={(event) => { const value = Number(event.target.value); setSat(value); push({ temp, bright, sat: value }) }} />
            <output>{sat}</output>
          </label>
        </div>
      ) : (
        <ColorTools page={page} onTint={(patch) => setLook(clip.id, patch)} />
      )}
    </aside>
  )
}
