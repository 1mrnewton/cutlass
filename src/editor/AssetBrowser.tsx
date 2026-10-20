import { useEffect, useRef, useState, type DragEvent as ReactDragEvent, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent } from "react"
import { ClipMenu, type MenuEntry } from "./ClipMenu"
import { Icon } from "./icons"
import { Tip } from "./Tooltip"
import { assetDragId, assetOnTimeline, assets, beginAssetDrag, endAssetDrag, formatTime, placementFor, stockFootage, type ProjectState, type ToolId } from "./model"

type Folder = { id: string; name: string; parentId: string | null }
type MenuState = { x: number; y: number; target: "empty" | "folder" | "file"; id?: string }

const emptyMenu: MenuEntry[] = [
  { type: "item", id: "new-folder", label: "New folder" },
  {
    type: "item",
    id: "sort",
    label: "Sort",
    children: [
      { type: "item", id: "sort-name", label: "Name" },
      { type: "item", id: "sort-duration", label: "Duration" },
    ],
  },
]

const folderMenu: MenuEntry[] = [
  { type: "item", id: "delete", label: "Delete" },
  { type: "item", id: "open-file", label: "Open file location", disabled: true },
  { type: "item", id: "new-folder-with", label: "New folder with selected item", disabled: true },
  { type: "item", id: "rename", label: "Rename" },
  { type: "item", id: "link-media", label: "Link to media", disabled: true },
  { type: "item", id: "export", label: "Export selected clips" },
]

const fileMenu: MenuEntry[] = [
  { type: "item", id: "open-file", label: "Open file location" },
  { type: "item", id: "delete", label: "Delete" },
  { type: "item", id: "new-folder-with", label: "New folder with selected item" },
  { type: "item", id: "link-media", label: "Link to media", disabled: true },
  { type: "item", id: "split-scenes", label: "Split scenes", badge: "Free" },
  { type: "item", id: "export", label: "Export selected clips" },
]

export function AssetBrowser({
  tool,
  project,
  onCommand,
  onNote,
  onInsert,
}: {
  tool: ToolId
  project: ProjectState
  onCommand: (command: string) => void
  onNote: (note: string) => void
  onInsert: (id: string) => void
}) {
  const [query, setQuery] = useState("")
  const [selected, setSelected] = useState<string[]>([])
  const [source, setSource] = useState<"project" | "stock">("project")
  const [folders, setFolders] = useState<Folder[]>([])
  const [places, setPlaces] = useState<Record<string, string | null>>({})
  const [hidden, setHidden] = useState<string[]>([])
  const [sort, setSort] = useState<"name" | "duration">("name")
  const [folderId, setFolderId] = useState<string | null>(null)
  const [renameId, setRenameId] = useState<string | null>(null)
  const [menu, setMenu] = useState<MenuState | null>(null)
  const [dropTarget, setDropTarget] = useState<string | null>(null)
  const [marquee, setMarquee] = useState<{ left: number; top: number; width: number; height: number } | null>(null)
  const dragged = useRef(false)
  const dragGhost = useRef<HTMLCanvasElement | null>(null)
  const skipClickUntil = useRef(0)
  const selectedRef = useRef(selected)
  const anchor = useRef<string | null>(null)
  selectedRef.current = selected
  const stockShelf = source === "stock" && tool === "media"
  const bin = tool === "media" && !stockShelf
  const needle = query.trim().toLowerCase()

  useEffect(() => {
    if (!renameId) return
    const field = document.getElementById(`rename-${renameId}`)
    if (field instanceof HTMLInputElement) {
      field.focus()
      field.select()
    }
  }, [renameId])

  const shown = stockShelf
    ? stockFootage
        .filter((shot) => !needle || shot.name.toLowerCase().includes(needle))
        .map((shot) => ({
          id: shot.id,
          name: shot.name,
          meta: `Free · ${formatTime(shot.duration)}`,
          swatch: shot.swatch,
        }))
    : assets
        .filter((asset) => asset.tool === tool)
        .filter((asset) => !needle || asset.name.toLowerCase().includes(needle))
        .filter((asset) => {
          if (!bin) return true
          if (hidden.includes(asset.id)) return false
          return (places[asset.id] ?? null) === folderId
        })
        .map((asset) => ({
          id: asset.id,
          name: asset.name,
          meta: asset.meta,
          swatch: asset.swatch,
        }))

  const visibleFolders = bin
    ? folders
        .filter((folder) => folder.parentId === folderId)
        .filter((folder) => !needle || folder.name.toLowerCase().includes(needle))
        .sort((a, b) => a.name.localeCompare(b.name))
    : []

  const files = bin
    ? [...shown].sort((a, b) => (sort === "name" ? a.name.localeCompare(b.name) : durationOf(a.meta) - durationOf(b.meta)))
    : shown

  const trail = folderTrail(folders, folderId)
  const orderRef = useRef<string[]>([])
  orderRef.current = [...visibleFolders.map((folder) => folder.id), ...files.map((file) => file.id)]

  const chooseMedia = (id: string, event: { shiftKey: boolean; metaKey: boolean; ctrlKey: boolean }) => {
    const order = orderRef.current
    const toggle = event.metaKey || event.ctrlKey
    const anchorId = anchor.current
    if (event.shiftKey && anchorId) {
      const from = order.indexOf(anchorId)
      const to = order.indexOf(id)
      if (from >= 0 && to >= 0) {
        const [start, end] = from < to ? [from, to] : [to, from]
        const range = order.slice(start, end + 1)
        setSelected(toggle ? [...new Set([...selectedRef.current, ...range])] : range)
        return
      }
    }
    if (toggle) {
      const current = selectedRef.current
      setSelected(current.includes(id) ? current.filter((item) => item !== id) : [...current, id])
      anchor.current = id
      return
    }
    setSelected([id])
    anchor.current = id
  }

  const onGridPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (tool !== "media" || event.button !== 0) return
    const target = event.target
    if (!(target instanceof Element) || target.closest(".card, .card-add")) return
    const grid = event.currentTarget
    const originX = event.clientX
    const originY = event.clientY
    const shift = event.shiftKey
    const base = shift ? selectedRef.current : []
    let moving = false
    try {
      grid.setPointerCapture(event.pointerId)
    } catch {
      // Pointer capture can fail; the window listeners still follow the drag.
    }
    const boxOf = (clientX: number, clientY: number) => {
      const bounds = grid.getBoundingClientRect()
      return {
        left: Math.min(originX, clientX) - bounds.left + grid.scrollLeft,
        top: Math.min(originY, clientY) - bounds.top + grid.scrollTop,
        width: Math.abs(clientX - originX),
        height: Math.abs(clientY - originY),
      }
    }
    const hits = (clientX: number, clientY: number) => {
      const left = Math.min(originX, clientX)
      const top = Math.min(originY, clientY)
      const right = Math.max(originX, clientX)
      const bottom = Math.max(originY, clientY)
      return [...grid.querySelectorAll<HTMLElement>(".card")].flatMap((card) => {
        const id = card.dataset.assetId
        if (!id) return []
        const rect = card.getBoundingClientRect()
        if (rect.right < left || rect.left > right || rect.bottom < top || rect.top > bottom) return []
        return [id]
      })
    }
    const move = (pointerEvent: PointerEvent) => {
      if (!moving && Math.hypot(pointerEvent.clientX - originX, pointerEvent.clientY - originY) < 4) return
      moving = true
      setMarquee(boxOf(pointerEvent.clientX, pointerEvent.clientY))
    }
    const up = (pointerEvent: PointerEvent) => {
      grid.removeEventListener("pointermove", move)
      grid.removeEventListener("pointerup", up)
      grid.removeEventListener("pointercancel", up)
      setMarquee(null)
      if (!moving) {
        setSelected([])
        anchor.current = null
        return
      }
      const found = hits(pointerEvent.clientX, pointerEvent.clientY)
      const next = shift ? [...new Set([...base, ...found])] : found
      setSelected(next)
      anchor.current = next.at(-1) ?? anchor.current
      skipClickUntil.current = performance.now() + 300
    }
    grid.addEventListener("pointermove", move)
    grid.addEventListener("pointerup", up)
    grid.addEventListener("pointercancel", up)
  }

  const dragFile = () => {
    const id = assetDragId()
    if (!id || !assets.some((asset) => asset.id === id && asset.tool === "media")) return null
    return id
  }

  const acceptMove = (parent: string | null) => {
    const id = dragFile()
    return id !== null && (places[id] ?? null) !== parent
  }

  const bindDrop = (parent: string | null) => {
    const key = parent ?? "all"
    return {
      "data-drop": dropTarget === key ? "true" : "false",
      onDragEnter: (event: ReactDragEvent) => {
        if (!acceptMove(parent)) return
        event.preventDefault()
        setDropTarget(key)
      },
      onDragOver: (event: ReactDragEvent) => {
        if (!acceptMove(parent)) return
        event.preventDefault()
        event.dataTransfer.dropEffect = "move"
        setDropTarget(key)
      },
      onDragLeave: (event: ReactDragEvent) => {
        const next = event.relatedTarget
        if (next instanceof Node && event.currentTarget.contains(next)) return
        setDropTarget((current) => (current === key ? null : current))
      },
      onDrop: (event: ReactDragEvent) => {
        const id = dragFile()
        setDropTarget(null)
        if (!id || (places[id] ?? null) === parent) return
        event.preventDefault()
        event.stopPropagation()
        skipClickUntil.current = performance.now() + 300
        setPlaces((current) => ({ ...current, [id]: parent }))
        const name = assets.find((asset) => asset.id === id)?.name ?? "Clip"
        const where = parent ? (folders.find((folder) => folder.id === parent)?.name ?? "Folder") : "All"
        onNote(`Moved ${name} to ${where}`)
      },
    }
  }

  const openAfterClick = (go: () => void) => {
    if (performance.now() < skipClickUntil.current) return
    go()
  }

  const openMenu = (event: ReactMouseEvent, next: MenuState) => {
    event.preventDefault()
    event.stopPropagation()
    setMenu(next)
  }

  const nextFolderName = () => {
    const used = new Set(folders.map((folder) => folder.name))
    if (!used.has("Untitled folder")) return "Untitled folder"
    let index = 2
    while (used.has(`Untitled folder ${index}`)) index += 1
    return `Untitled folder ${index}`
  }

  const addFolder = (fileId?: string) => {
    const id = `folder-${crypto.randomUUID()}`
    const name = nextFolderName()
    setFolders((current) => [...current, { id, name, parentId: folderId }])
    if (fileId) setPlaces((current) => ({ ...current, [fileId]: id }))
    setSelected([id])
    anchor.current = id
    setRenameId(id)
  }

  const onPick = (id: string) => {
    const target = menu
    setMenu(null)
    if (!target) return
    if (id === "new-folder") {
      addFolder()
      return
    }
    if (id === "sort-name" || id === "sort-duration") {
      setSort(id === "sort-name" ? "name" : "duration")
      onNote(id === "sort-name" ? "Sorted by name" : "Sorted by duration")
      return
    }
    if (id === "export") {
      onNote("The prototype does not write a file.")
      return
    }
    if (id === "open-file") {
      onNote("Library cards are placeholders.")
      return
    }
    if (id === "split-scenes") {
      onNote("Split scenes")
      return
    }
    if (id === "rename" && target.id) {
      setRenameId(target.id)
      return
    }
    if (id === "new-folder-with" && target.id) {
      addFolder(target.id)
      return
    }
    if (id === "delete" && target.target === "file" && target.id) {
      const name = files.find((file) => file.id === target.id)?.name ?? "Clip"
      setHidden((current) => [...current, target.id!])
      setSelected((current) => current.filter((item) => item !== target.id))
      onNote(`Deleted ${name}`)
      return
    }
    if (id === "delete" && target.target === "folder" && target.id) {
      const folder = folders.find((item) => item.id === target.id)
      const gone = folderIdsUnder(folders, target.id)
      setFolders((current) => current.filter((item) => !gone.has(item.id)))
      setPlaces((current) => {
        const next = { ...current }
        for (const [file, place] of Object.entries(next)) {
          if (place && gone.has(place)) next[file] = folder?.parentId ?? null
        }
        return next
      })
      if (folderId && gone.has(folderId)) setFolderId(folder?.parentId ?? null)
      setSelected((current) => current.filter((item) => !gone.has(item)))
      onNote(`Deleted ${folder?.name ?? "Folder"}`)
    }
  }

  const cards = files.map((asset) => {
    const source = placementFor(asset.id)
    const placeable = source?.kind != null
    const card = (
    <button
      key={asset.id}
      type="button"
      className="card"
      data-tip={`${asset.name} · ${asset.meta}`}
      draggable={placeable}
      data-asset-id={asset.id}
      data-picked={selected.includes(asset.id) ? "true" : "false"}
      style={{ background: asset.swatch }}
      onContextMenu={
        bin
          ? (event) => {
              if (!selectedRef.current.includes(asset.id)) {
                setSelected([asset.id])
                anchor.current = asset.id
              }
              openMenu(event, { x: event.clientX, y: event.clientY, target: "file", id: asset.id })
            }
          : undefined
      }
      onDragStart={(event) => {
        if (event.shiftKey || event.metaKey || event.ctrlKey) {
          event.preventDefault()
          return
        }
        if (!placeable) {
          event.preventDefault()
          return
        }
        dragged.current = true
        event.dataTransfer.setData("text/plain", asset.id)
        event.dataTransfer.effectAllowed = "copyMove"
        dragGhost.current?.remove()
        const ghost = cardDragImage(asset.name, asset.meta, asset.swatch)
        document.body.append(ghost)
        dragGhost.current = ghost
        const rect = event.currentTarget.getBoundingClientRect()
        event.dataTransfer.setDragImage(ghost, event.clientX - rect.left, event.clientY - rect.top)
        beginAssetDrag(asset.id)
      }}
      onDragEnd={() => {
        dragGhost.current?.remove()
        dragGhost.current = null
        endAssetDrag()
        setDropTarget(null)
        window.setTimeout(() => {
          dragged.current = false
        }, 0)
      }}
      onClick={(event) => {
        if (dragged.current) return
        if (tool === "media") {
          chooseMedia(asset.id, event)
          return
        }
        setSelected([asset.id])
        if (asset.id === "wedding") onCommand("Using all these footages, make a wedding film")
      }}
      onDoubleClick={() => {
        if (stockShelf) onCommand(`stock ${asset.id}`)
      }}
    >
      {placeable && assetOnTimeline(project, asset.id) && (
        <span className="card-used" data-tip="On the timeline">On timeline</span>
      )}
      <span className="thumb" style={{ background: asset.swatch }} />
      <strong>{asset.name}</strong>
      <small>{asset.meta}</small>
    </button>
    )
    if (!placeable) return card
    return (
      <div className="card-wrap" key={asset.id}>
        {card}
        <span className="card-add-host">
          <Tip label="Add to timeline">
            <button type="button" className="card-add" aria-label="Add to timeline" onClick={() => onInsert(asset.id)}>
              <Icon name="plus" size={14} />
            </button>
          </Tip>
        </span>
      </div>
    )
  })

  const menuItems = menu?.target === "folder" ? folderMenu : menu?.target === "file" ? fileMenu : emptyMenu

  return (
    <div className="browser">
      <input
        className="asset-search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder={stockShelf ? "Search free stock" : `Search ${tool}`}
        aria-label={stockShelf ? "Search free stock" : `Search ${tool}`}
      />
      {tool === "media" && (
        <div className="media-source" role="radiogroup" aria-label="Media source">
          <button
            type="button"
            role="radio"
            aria-checked={source === "project"}
            data-active={source === "project" ? "true" : "false"}
            onClick={() => setSource("project")}
          >
            Project
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={source === "stock"}
            data-active={source === "stock" ? "true" : "false"}
            onClick={() => setSource("stock")}
          >
            Free stock
          </button>
        </div>
      )}
      {bin && (
        <div className="bin-path">
          <button type="button" data-current={folderId === null ? "true" : "false"} {...bindDrop(null)} onClick={() => openAfterClick(() => setFolderId(null))}>
            All
          </button>
          {trail.map((folder) => (
            <span key={folder.id}>
              <span aria-hidden="true">/</span>
              <button
                type="button"
                data-current={folder.id === folderId ? "true" : "false"}
                {...bindDrop(folder.id)}
                onClick={() => openAfterClick(() => setFolderId(folder.id))}
              >
                {folder.name}
              </button>
            </span>
          ))}
        </div>
      )}
      <div
        className="asset-grid"
        onPointerDown={onGridPointerDown}
        onContextMenu={bin ? (event) => openMenu(event, { x: event.clientX, y: event.clientY, target: "empty" }) : undefined}
      >
        {visibleFolders.map((folder) => (
          <button
            key={folder.id}
            type="button"
            className="card"
            data-tip={folder.name}
            data-asset-id={folder.id}
            data-picked={selected.includes(folder.id) ? "true" : "false"}
            {...bindDrop(folder.id)}
            onContextMenu={(event) => {
              if (!selectedRef.current.includes(folder.id)) {
                setSelected([folder.id])
                anchor.current = folder.id
              }
              openMenu(event, { x: event.clientX, y: event.clientY, target: "folder", id: folder.id })
            }}
            onClick={(event) => {
              if (renameId === folder.id) return
              if (event.shiftKey || event.metaKey || event.ctrlKey) {
                chooseMedia(folder.id, event)
                return
              }
              openAfterClick(() => {
                setFolderId(folder.id)
                setSelected([folder.id])
                anchor.current = folder.id
              })
            }}
          >
            <span className="thumb folder-thumb">
              <Icon name="folder" size={40} />
            </span>
            {renameId === folder.id ? (
              <input
                id={`rename-${folder.id}`}
                className="bin-rename"
                defaultValue={folder.name}
                aria-label="Folder name"
                onClick={(event) => event.stopPropagation()}
                onKeyDown={(event) => {
                  if (event.key === "Enter") event.currentTarget.blur()
                  if (event.key === "Escape") {
                    event.currentTarget.value = folder.name
                    setRenameId(null)
                  }
                }}
                onBlur={(event) => {
                  const name = event.target.value.trim() || folder.name
                  setFolders((current) => current.map((item) => (item.id === folder.id ? { ...item, name } : item)))
                  setRenameId(null)
                }}
              />
            ) : (
              <strong>{folder.name}</strong>
            )}
          </button>
        ))}
        {cards.length > 0 ? cards : needle ? <p className="empty">Nothing under that name.</p> : null}
        {marquee && <div className="marquee" style={marquee} />}
      </div>
      {menu && <ClipMenu x={menu.x} y={menu.y} items={menuItems} onClose={() => setMenu(null)} onPick={onPick} />}
    </div>
  )
}

function cardDragImage(name: string, meta: string, swatch: string) {
  const width = 144
  const height = 96
  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = height
  canvas.style.position = "fixed"
  canvas.style.left = "-1000px"
  canvas.style.top = "0"
  canvas.style.pointerEvents = "none"
  const ctx = canvas.getContext("2d")
  if (!ctx) return canvas
  const radius = 12
  ctx.clearRect(0, 0, width, height)
  ctx.beginPath()
  ctx.moveTo(radius, 0)
  ctx.arcTo(width, 0, width, height, radius)
  ctx.arcTo(width, height, 0, height, radius)
  ctx.arcTo(0, height, 0, 0, radius)
  ctx.arcTo(0, 0, width, 0, radius)
  ctx.closePath()
  ctx.fillStyle = swatch
  ctx.fill()
  const family = getComputedStyle(document.body).fontFamily
  ctx.shadowColor = "rgba(0, 0, 0, 0.55)"
  ctx.shadowBlur = 2
  ctx.shadowOffsetY = 1
  ctx.fillStyle = "#fff"
  ctx.font = `600 12px ${family}`
  ctx.textBaseline = "top"
  ctx.fillText(name, 8, 60)
  ctx.font = `11px ${family}`
  ctx.fillStyle = "rgba(255, 255, 255, 0.86)"
  ctx.fillText(meta, 8, 76)
  return canvas
}

function durationOf(meta: string) {
  const match = meta.match(/(\d+):(\d+)/)
  if (!match) return Number.MAX_SAFE_INTEGER
  return Number(match[1]) * 60 + Number(match[2])
}

function folderIdsUnder(folders: Folder[], id: string) {
  const gone = new Set<string>([id])
  let grew = true
  while (grew) {
    grew = false
    for (const folder of folders) {
      if (folder.parentId && gone.has(folder.parentId) && !gone.has(folder.id)) {
        gone.add(folder.id)
        grew = true
      }
    }
  }
  return gone
}

function folderTrail(folders: Folder[], folderId: string | null) {
  const trail: Folder[] = []
  let current = folders.find((folder) => folder.id === folderId) ?? null
  while (current) {
    trail.unshift(current)
    current = folders.find((folder) => folder.id === current?.parentId) ?? null
  }
  return trail
}
