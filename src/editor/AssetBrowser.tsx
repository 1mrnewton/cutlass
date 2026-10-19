import { useState } from "react"
import { assets, type ToolId } from "./model"

export function AssetBrowser({
  tool,
  layout,
  onCommand,
}: {
  tool: ToolId
  layout: "grid" | "row"
  onCommand: (command: string) => void
}) {
  const [query, setQuery] = useState("")
  const [picked, setPicked] = useState<string | null>(null)
  const shown = assets.filter((asset) => {
    if (asset.tool !== tool) return false
    if (layout === "row" || !query.trim()) return true
    return asset.name.toLowerCase().includes(query.trim().toLowerCase())
  })

  const cards = shown.map((asset) => (
    <button
      key={asset.id}
      type="button"
      className="card"
      data-picked={picked === asset.id ? "true" : "false"}
      onClick={() => {
        setPicked(asset.id)
        if (asset.id === "wedding") {
          onCommand("Using all these footages, make a wedding film")
        }
      }}
    >
      <span className="thumb" style={{ background: asset.swatch }} />
      <strong>{asset.name}</strong>
      <small>{asset.meta}</small>
    </button>
  ))

  if (layout === "row") {
    return <div className="asset-row">{cards}</div>
  }

  return (
    <div className="browser">
      <input
        className="asset-search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder={`Search ${tool}`}
        aria-label={`Search ${tool}`}
      />
      <div className="asset-grid">
        {cards.length > 0 ? cards : <p className="empty">Nothing under that name.</p>}
      </div>
    </div>
  )
}
