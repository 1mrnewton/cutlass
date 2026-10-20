import { useEffect, useId, useRef, useState, type ReactNode } from "react"
import { formatTime, scenes } from "./model"

const resolutions = ["144p", "240p", "360p", "480p", "720p", "1080p", "1440p", "4K", "8K"] as const
const rates = ["24", "30", "60"] as const
const qualities = ["Standard", "High", "Best"] as const
const formats = ["MP4", "MOV", "WebM"] as const
const rateControls = ["VBR", "CBR"] as const
const profiles = ["Proxy", "LT", "422", "HQ"] as const
const sampleRates = ["44.1 kHz", "48 kHz"] as const
const channels = ["Stereo", "Mono"] as const
const audioRates = ["128", "192", "320"] as const
const gifRates = ["10", "15", "24"] as const
const gifLoops = ["Once", "Loop"] as const
const captionFiles = ["Burn in", "SRT"] as const

type Resolution = (typeof resolutions)[number]
type Quality = (typeof qualities)[number]
type Format = (typeof formats)[number]
type VideoCodec = "H.264" | "H.265" | "ProRes" | "VP9" | "AV1"
type AudioCodec = "AAC" | "PCM" | "Opus"

const videoCodecs: Record<Format, readonly VideoCodec[]> = {
  MP4: ["H.264", "H.265"],
  MOV: ["H.264", "H.265", "ProRes"],
  WebM: ["VP9", "AV1"],
}

const audioCodecs: Record<Format, readonly AudioCodec[]> = {
  MP4: ["AAC"],
  MOV: ["AAC", "PCM"],
  WebM: ["Opus"],
}

const suggestedBitrate: Record<Resolution, Record<Quality, string>> = {
  "144p": { Standard: "0.2", High: "0.4", Best: "0.8" },
  "240p": { Standard: "0.4", High: "0.8", Best: "1.5" },
  "360p": { Standard: "0.8", High: "1.5", Best: "2.5" },
  "480p": { Standard: "1.5", High: "2.5", Best: "4" },
  "720p": { Standard: "5", High: "8", Best: "16" },
  "1080p": { Standard: "8", High: "16", Best: "40" },
  "1440p": { Standard: "12", High: "24", Best: "50" },
  "4K": { Standard: "20", High: "45", Best: "80" },
  "8K": { Standard: "50", High: "100", Best: "160" },
}

export function ExportDialog({
  duration,
  name,
  onClose,
}: {
  duration: number
  name: string
  onClose: () => void
}) {
  const titleId = useId()
  const nameRef = useRef<HTMLInputElement>(null)
  const [exportName, setName] = useState(name)
  const [resolution, setResolution] = useState<Resolution>("1080p")
  const [rate, setRate] = useState<(typeof rates)[number]>("30")
  const [quality, setQuality] = useState<Quality>("High")
  const [format, setFormat] = useState<Format>("MP4")
  const [codec, setCodec] = useState<VideoCodec>("H.264")
  const [rateControl, setRateControl] = useState<(typeof rateControls)[number]>("VBR")
  const [profile, setProfile] = useState<(typeof profiles)[number]>("422")
  const [bitrate, setBitrate] = useState("16")
  const [bitrateCustom, setBitrateCustom] = useState(false)
  const [videoOn, setVideoOn] = useState(true)
  const [audioOn, setAudioOn] = useState(true)
  const [gifOn, setGifOn] = useState(false)
  const [gifRate, setGifRate] = useState<(typeof gifRates)[number]>("15")
  const [gifLoop, setGifLoop] = useState<(typeof gifLoops)[number]>("Loop")
  const [captionOn, setCaptionOn] = useState(false)
  const [captionFile, setCaptionFile] = useState<(typeof captionFiles)[number]>("SRT")
  const [audioCodec, setAudioCodec] = useState<AudioCodec>("AAC")
  const [sampleRate, setSampleRate] = useState<(typeof sampleRates)[number]>("48 kHz")
  const [channel, setChannel] = useState<(typeof channels)[number]>("Stereo")
  const [audioBitrate, setAudioBitrate] = useState<(typeof audioRates)[number]>("192")

  useEffect(() => {
    nameRef.current?.focus()
    nameRef.current?.select()
    const onKey = (event: KeyboardEvent) => {
      if (event.code === "Escape") {
        event.preventDefault()
        onClose()
      }
      if (event.code === "Space") {
        const target = event.target
        if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return
        event.preventDefault()
        event.stopPropagation()
      }
    }
    window.addEventListener("keydown", onKey, true)
    return () => window.removeEventListener("keydown", onKey, true)
  }, [onClose])

  const prores = codec === "ProRes"
  const wash = scenes.vows

  const applyQuality = (next: Quality) => {
    setQuality(next)
    setBitrate(suggestedBitrate[resolution][next])
    setBitrateCustom(false)
  }

  const applyResolution = (next: Resolution) => {
    setResolution(next)
    if (!bitrateCustom) setBitrate(suggestedBitrate[next][quality])
  }

  const applyFormat = (next: Format) => {
    setFormat(next)
    const videos = videoCodecs[next]
    if (!videos.includes(codec)) setCodec(videos[0])
    const audios = audioCodecs[next]
    if (!audios.includes(audioCodec)) setAudioCodec(audios[0])
  }

  return (
    <div className="scrim" onClick={onClose}>
      <div
        className="export-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="dialog-head">
          <button type="button" className="dialog-close" onClick={onClose} aria-label="Close">
            ×
          </button>
          <h2 id={titleId}>Export</h2>
        </header>

        <div className="export-body">
          <div className="cover-col">
            <div
              className="cover"
              style={{ background: `linear-gradient(155deg, ${wash[0]}, ${wash[1]} 46%, ${wash[2]})` }}
            >
              <p>{exportName || "Untitled"}</p>
            </div>
          </div>

          <div className="settings">
            <label className="setting">
              <span>Name</span>
              <input
                ref={nameRef}
                value={exportName}
                onChange={(event) => setName(event.target.value)}
              />
            </label>
            <Section label="Video" on={videoOn} onToggle={() => setVideoOn((value) => !value)}>
              <Choice label="Resolution" value={resolution} options={resolutions} onChange={applyResolution} />
              <Choice label="Frame rate" value={rate} options={rates} onChange={setRate} />
              <Choice label="Format" value={format} options={formats} onChange={applyFormat} />
              <Choice label="Codec" value={codec} options={videoCodecs[format]} onChange={setCodec} />
              {prores ? (
                <Choice label="ProRes" value={profile} options={profiles} onChange={setProfile} />
              ) : (
                <>
                  <Choice label="Quality" value={quality} options={qualities} onChange={applyQuality} />
                  <Choice label="Rate" value={rateControl} options={rateControls} onChange={setRateControl} />
                  <label className="setting">
                    <span>Bitrate</span>
                    <div className="unit-field">
                      <input
                        inputMode="decimal"
                        value={bitrate}
                        aria-label="Video bitrate"
                        onChange={(event) => {
                          setBitrate(event.target.value.replace(/[^\d.]/g, ""))
                          setBitrateCustom(true)
                        }}
                      />
                      <span>Mbps</span>
                    </div>
                  </label>
                </>
              )}
            </Section>
            <Section label="Audio" on={audioOn} onToggle={() => setAudioOn((value) => !value)}>
              <Choice label="Codec" value={audioCodec} options={audioCodecs[format]} onChange={setAudioCodec} />
              <Choice label="Sample rate" value={sampleRate} options={sampleRates} onChange={setSampleRate} />
              <Choice label="Channels" value={channel} options={channels} onChange={setChannel} />
              {audioCodec !== "PCM" && (
                <Choice label="kbps" value={audioBitrate} options={audioRates} onChange={setAudioBitrate} />
              )}
            </Section>
            <Section label="GIF" on={gifOn} startOpen={false} onToggle={() => setGifOn((value) => !value)}>
              <Choice label="Frame rate" value={gifRate} options={gifRates} onChange={setGifRate} />
              <Choice label="Loop" value={gifLoop} options={gifLoops} onChange={setGifLoop} />
            </Section>
            <Section label="Caption" on={captionOn} startOpen={false} onToggle={() => setCaptionOn((value) => !value)}>
              <Choice label="File" value={captionFile} options={captionFiles} onChange={setCaptionFile} />
            </Section>
          </div>
        </div>

        <footer className="dialog-actions">
          <span className="dialog-duration">{formatTime(duration)}</span>
          <button type="button" className="text-btn" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="export" onClick={onClose}>
            Export
          </button>
        </footer>
      </div>
    </div>
  )
}

function Section({
  label,
  on,
  onToggle,
  startOpen = true,
  children,
}: {
  label: string
  on: boolean
  onToggle: () => void
  startOpen?: boolean
  children: ReactNode
}) {
  const [open, setOpen] = useState(startOpen)
  return (
    <section>
      <div className="setting group-row">
        <button
          type="button"
          className="section-toggle"
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
        >
          <span className="chevron" data-open={open ? "true" : "false"} />
          <span className="eyebrow">{label}</span>
        </button>
        <button
          type="button"
          className="switch"
          aria-pressed={on}
          aria-label={on ? `${label} on` : `${label} off`}
          onClick={onToggle}
        >
          <span />
        </button>
      </div>
      {open && children}
    </section>
  )
}

function Choice<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: readonly T[]
  onChange: (value: T) => void
}) {
  return (
    <label className="setting">
      <span>{label}</span>
      <span className="select">
        <select
          aria-label={label}
          value={value}
          onChange={(event) => onChange(event.target.value as T)}
        >
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </span>
    </label>
  )
}
