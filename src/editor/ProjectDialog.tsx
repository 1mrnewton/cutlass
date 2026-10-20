import { useEffect, useId, useRef, useState } from "react"
import {
  aspectRatios,
  colorSpaces,
  frameRateLabel,
  frameRates,
  projectPath,
  projectResolutions,
  supportsDropFrame,
  type FrameRate,
  type ProjectSettings,
} from "./model"

export function ProjectDialog({
  settings,
  onClose,
  onSave,
}: {
  settings: ProjectSettings
  onClose: () => void
  onSave: (settings: ProjectSettings) => void
}) {
  const titleId = useId()
  const nameRef = useRef<HTMLInputElement>(null)
  const [draft, setDraft] = useState(settings)
  const patch = (next: Partial<ProjectSettings>) => setDraft((current) => ({ ...current, ...next }))

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

  const save = () => {
    const name = draft.name.trim() || "Untitled"
    onSave({ ...draft, name })
  }

  return (
    <div className="scrim" onClick={onClose}>
      <div
        className="export-dialog project-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="dialog-head">
          <button type="button" className="dialog-close" onClick={onClose} aria-label="Close">
            ×
          </button>
          <h2 id={titleId}>Project settings</h2>
        </header>

        <div className="project-body">
          <div className="settings">
            <p className="eyebrow">Details</p>
            <label className="setting">
              <span>Name</span>
              <input ref={nameRef} aria-label="Name" value={draft.name} onChange={(event) => patch({ name: event.target.value })} />
            </label>
            <div className="setting">
              <span>Save to</span>
              <p className="setting-value">{projectPath(draft.name)}</p>
            </div>
            <div className="setting" role="radiogroup" aria-label="Import material">
              <span>Import material</span>
              <div className="radio-stack">
                <label>
                  <input
                    type="radio"
                    name="import-material"
                    checked={draft.importMaterial === "copy"}
                    onChange={() => patch({ importMaterial: "copy" })}
                  />
                  Copy to project
                </label>
                <label>
                  <input
                    type="radio"
                    name="import-material"
                    checked={draft.importMaterial === "keep"}
                    onChange={() => patch({ importMaterial: "keep" })}
                  />
                  Keep in original place
                </label>
              </div>
            </div>
            <Choice label="Color space" value={draft.colorSpace} options={colorSpaces} onChange={(colorSpace) => patch({ colorSpace })} />
            <div className="setting">
              <span>Free layer</span>
              <button
                type="button"
                className="switch"
                aria-pressed={draft.freeLayer}
                aria-label={draft.freeLayer ? "Free layer on" : "Free layer off"}
                disabled={draft.freeLayer}
                onClick={() => patch({ freeLayer: true })}
              >
                <span />
              </button>
            </div>
            <p className="lock-note">Cannot be turned off after it is turned on.</p>
            <div className="setting">
              <span>Proxy</span>
              <button
                type="button"
                className="switch"
                aria-pressed={draft.proxy}
                aria-label={draft.proxy ? "Proxy on" : "Proxy off"}
                onClick={() => patch({ proxy: !draft.proxy })}
              >
                <span />
              </button>
            </div>

            <p className="eyebrow">Timeline</p>
            <Choice label="Ratio" value={draft.ratio} options={aspectRatios} onChange={(ratio) => patch({ ratio })} />
            <Choice label="Resolution" value={draft.resolution} options={projectResolutions} onChange={(resolution) => patch({ resolution })} />
            <Choice
              label="Frame rate"
              value={draft.frameRate}
              options={frameRates}
              labelFor={(rate) => frameRateLabel(rate)}
              onChange={(frameRate) => patch({ frameRate, dropFrame: dropFrameFor(frameRate, draft) })}
            />
            <div className="setting">
              <span>Drop frame</span>
              <button
                type="button"
                className="switch"
                aria-pressed={draft.dropFrame}
                aria-label={draft.dropFrame ? "Drop frame on" : "Drop frame off"}
                disabled={!supportsDropFrame(draft.frameRate)}
                onClick={() => patch({ dropFrame: !draft.dropFrame })}
              >
                <span />
              </button>
            </div>
            <p className="field-note">29.97 drops 2 frame numbers a minute. 59.94 drops 4. Every tenth minute keeps them.</p>
          </div>
        </div>

        <footer className="dialog-actions">
          <button type="button" className="text-btn" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="export" onClick={save}>
            Save
          </button>
        </footer>
      </div>
    </div>
  )
}

function dropFrameFor(frameRate: FrameRate, draft: ProjectSettings) {
  if (!supportsDropFrame(frameRate)) return false
  if (supportsDropFrame(draft.frameRate)) return draft.dropFrame
  return true
}

function Choice<T extends string>({
  label,
  value,
  options,
  labelFor,
  onChange,
}: {
  label: string
  value: T
  options: readonly T[]
  labelFor?: (value: T) => string
  onChange: (value: T) => void
}) {
  return (
    <label className="setting">
      <span>{label}</span>
      <span className="select">
        <select aria-label={label} value={value} onChange={(event) => onChange(event.target.value as T)}>
          {options.map((option) => (
            <option key={option} value={option}>
              {labelFor ? labelFor(option) : option}
            </option>
          ))}
        </select>
      </span>
    </label>
  )
}
