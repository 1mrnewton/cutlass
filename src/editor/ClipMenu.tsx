import { useEffect, useLayoutEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Icon } from "./icons"
import type { TrackKind } from "./model"

export type MenuEntry =
  | { type: "sep" }
  | {
      type: "item"
      id: string
      label: string
      shortcut?: string
      badge?: string
      disabled?: boolean
      children?: MenuEntry[]
    }

const copyCutDelete: MenuEntry[] = [
  { type: "item", id: "copy", label: "Copy", shortcut: "⌘C" },
  { type: "item", id: "cut", label: "Cut", shortcut: "⌘X" },
  { type: "item", id: "delete", label: "Delete", shortcut: "⌫" },
]

const tail: MenuEntry[] = [
  { type: "sep" },
  { type: "item", id: "deactivate", label: "Deactivate" },
  { type: "item", id: "replace", label: "Replace" },
]

export function menuFor(kind: TrackKind): MenuEntry[] {
  if (kind === "video") return videoMenu
  if (kind === "audio" || kind === "text_to_video_text_audio") return audioMenu
  if (kind === "text" || kind === "text_to_video_text") return textMenu
  if (kind === "sticker") return stickerMenu
  if (kind === "effect" || kind === "plugin_effect") return effectMenu
  if (kind === "filter") return filterMenu
  if (kind === "adjust") return adjustMenu
  if (kind === "composition") return compositionMenu
  return adMenu
}

const videoMenu: MenuEntry[] = [
  ...copyCutDelete,
  { type: "sep" },
  {
    type: "item",
    id: "edit",
    label: "Edit",
    children: [
      { type: "item", id: "edit-speed", label: "Speed" },
      { type: "item", id: "edit-animation", label: "Animation" },
      { type: "item", id: "edit-crop", label: "Crop" },
      { type: "item", id: "edit-opacity", label: "Opacity" },
    ],
  },
  { type: "item", id: "split-scenes", label: "Split scenes" },
  { type: "item", id: "transcript", label: "Transcript" },
  {
    type: "item",
    id: "isolate-voice",
    label: "Isolate voice",
    children: [
      { type: "item", id: "isolate-keep-voice", label: "Keep voice" },
      { type: "item", id: "isolate-keep-background", label: "Keep background" },
    ],
  },
  { type: "item", id: "extract-audio", label: "Extract audio", shortcut: "⇧⌘S" },
  { type: "item", id: "sync", label: "Sync video and audio", disabled: true },
  { type: "sep" },
  { type: "item", id: "compound", label: "Create compound clip", shortcut: "⌥G" },
  { type: "item", id: "multicam", label: "Create multi-camera clip", disabled: true },
  { type: "item", id: "group", label: "Group", disabled: true },
  { type: "item", id: "ungroup", label: "Ungroup", disabled: true },
  { type: "sep" },
  { type: "item", id: "export", label: "Export selected clips" },
  { type: "item", id: "deactivate", label: "Deactivate clip", shortcut: "V" },
  { type: "item", id: "trim", label: "Trim clip", disabled: true },
  { type: "item", id: "replace", label: "Replace clip" },
  { type: "item", id: "link-media", label: "Link to media", disabled: true },
  { type: "item", id: "open-file", label: "Open file location" },
  { type: "sep" },
  { type: "item", id: "edit-effects", label: "Edit effects", disabled: true },
  { type: "item", id: "variable-speed", label: "Show variable speed animation", shortcut: "⌥K" },
  {
    type: "item",
    id: "range",
    label: "Range",
    children: [
      { type: "item", id: "range-in", label: "Set in" },
      { type: "item", id: "range-out", label: "Set out" },
    ],
  },
  {
    type: "item",
    id: "render",
    label: "Render",
    children: [
      { type: "item", id: "render-preview", label: "Preview" },
      { type: "item", id: "render-full", label: "Full" },
    ],
  },
]

const audioMenu: MenuEntry[] = [
  ...copyCutDelete,
  { type: "sep" },
  { type: "item", id: "volume", label: "Volume" },
  { type: "item", id: "speed", label: "Speed" },
  { type: "item", id: "fade", label: "Fade" },
  { type: "item", id: "beat", label: "Beat" },
  ...tail,
]

const textMenu: MenuEntry[] = [
  ...copyCutDelete,
  { type: "sep" },
  { type: "item", id: "edit-text", label: "Edit text" },
  { type: "item", id: "style", label: "Style" },
  ...tail,
]

const stickerMenu: MenuEntry[] = [
  ...copyCutDelete,
  { type: "sep" },
  { type: "item", id: "flip", label: "Flip" },
  ...tail,
]

const effectMenu: MenuEntry[] = [
  ...copyCutDelete,
  { type: "sep" },
  { type: "item", id: "edit-effect", label: "Edit effect" },
  ...tail,
]

const filterMenu: MenuEntry[] = [
  ...copyCutDelete,
  { type: "sep" },
  { type: "item", id: "edit-filter", label: "Edit filter" },
  ...tail,
]

const adjustMenu: MenuEntry[] = [
  ...copyCutDelete,
  { type: "sep" },
  { type: "item", id: "edit-adjust", label: "Edit adjustment" },
  ...tail,
]

const compositionMenu: MenuEntry[] = [
  ...copyCutDelete,
  { type: "sep" },
  { type: "item", id: "render-preview", label: "Render" },
  ...tail,
]

const adMenu: MenuEntry[] = [...copyCutDelete, ...tail]

export function ClipMenu({
  x,
  y,
  items,
  onClose,
  onPick,
}: {
  x: number
  y: number
  items: MenuEntry[]
  onClose: () => void
  onPick: (id: string, label: string) => void
}) {
  const rootRef = useRef<HTMLDivElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const subRef = useRef<HTMLDivElement>(null)
  const closeRef = useRef(onClose)
  const [sub, setSub] = useState<{ id: string; top: number; left: number; items: MenuEntry[] } | null>(null)

  useLayoutEffect(() => {
    closeRef.current = onClose
  })

  useLayoutEffect(() => {
    const el = menuRef.current
    if (!el) return
    const box = el.getBoundingClientRect()
    const left = Math.min(Math.max(8, x), window.innerWidth - box.width - 8)
    const top = Math.min(Math.max(8, y), window.innerHeight - box.height - 8)
    el.style.left = `${left}px`
    el.style.top = `${top}px`
  }, [x, y, items])

  useLayoutEffect(() => {
    const el = subRef.current
    if (!el) return
    const box = el.getBoundingClientRect()
    const left = Math.min(Math.max(8, box.left), window.innerWidth - box.width - 8)
    const top = Math.min(Math.max(8, box.top), window.innerHeight - box.height - 8)
    el.style.left = `${left}px`
    el.style.top = `${top}px`
  }, [sub])

  useEffect(() => {
    const onPointer = (event: PointerEvent) => {
      if (rootRef.current?.contains(event.target as Node)) return
      closeRef.current()
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeRef.current()
    }
    const onScroll = (event: Event) => {
      if (rootRef.current?.contains(event.target as Node)) return
      closeRef.current()
    }
    window.addEventListener("pointerdown", onPointer)
    window.addEventListener("keydown", onKey)
    window.addEventListener("scroll", onScroll, true)
    return () => {
      window.removeEventListener("pointerdown", onPointer)
      window.removeEventListener("keydown", onKey)
      window.removeEventListener("scroll", onScroll, true)
    }
  }, [])

  return createPortal(
    <div ref={rootRef}>
      <div ref={menuRef} className="clip-menu" role="menu" style={{ left: x, top: y }} onScroll={() => setSub(null)}>
        {items.map((entry, index) =>
          entry.type === "sep" ? (
            <div key={`sep-${index}`} className="clip-menu-sep" role="separator" />
          ) : (
            <MenuRow
              key={entry.id}
              entry={entry}
              open={sub?.id === entry.id}
              onOpen={(next) => setSub(next)}
              onPick={onPick}
            />
          ),
        )}
      </div>
      {sub && (
        <div ref={subRef} className="clip-menu" role="menu" style={{ top: sub.top, left: sub.left }}>
          {sub.items.map((child) =>
            child.type === "item" ? (
              <button key={child.id} type="button" role="menuitem" disabled={child.disabled} onClick={() => onPick(child.id, child.label)}>
                <span>{child.label}</span>
              </button>
            ) : null,
          )}
        </div>
      )}
    </div>,
    document.body,
  )
}

function MenuRow({
  entry,
  open,
  onOpen,
  onPick,
}: {
  entry: Extract<MenuEntry, { type: "item" }>
  open: boolean
  onOpen: (next: { id: string; top: number; left: number; items: MenuEntry[] } | null) => void
  onPick: (id: string, label: string) => void
}) {
  const rowRef = useRef<HTMLButtonElement>(null)

  return (
    <button
      ref={rowRef}
      type="button"
      role="menuitem"
      disabled={entry.disabled}
      aria-haspopup={entry.children ? "menu" : undefined}
      aria-expanded={entry.children ? open : undefined}
      onMouseEnter={() => {
        if (!entry.children) {
          onOpen(null)
          return
        }
        const box = rowRef.current?.getBoundingClientRect()
        if (!box) return
        const width = 248
        const flip = box.right + width > window.innerWidth - 8
        onOpen({
          id: entry.id,
          top: box.top - 6,
          left: flip ? box.left - width + 6 : box.right - 6,
          items: entry.children,
        })
      }}
      onClick={() => {
        if (entry.disabled || entry.children) return
        onPick(entry.id, entry.label)
      }}
    >
      <span className="clip-menu-label">
        {entry.label}
        {entry.badge ? <span className="clip-menu-badge">{entry.badge}</span> : null}
      </span>
      {entry.shortcut ? <span className="clip-menu-key">{entry.shortcut}</span> : null}
      {entry.children ? (
        <span className="clip-menu-chevron">
          <Icon name="chevron" size={14} />
        </span>
      ) : null}
    </button>
  )
}
