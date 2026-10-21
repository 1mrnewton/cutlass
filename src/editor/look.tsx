import { createContext, useContext, useState, type ReactNode } from "react"

export type Look = {
  volume?: number
  opacity?: number
  amount?: number
  temp?: number
  sat?: number
  bright?: number
  bubble?: string
  textEffect?: string
}

type LookApi = {
  looks: Record<string, Look>
  setLook: (id: string, patch: Partial<Look>) => void
}

const LookContext = createContext<LookApi>({ looks: {}, setLook: () => {} })

export function LookProvider({ children }: { children: ReactNode }) {
  const [looks, setLooks] = useState<Record<string, Look>>({})
  const setLook = (id: string, patch: Partial<Look>) => {
    setLooks((current) => ({ ...current, [id]: { ...current[id], ...patch } }))
  }
  return <LookContext.Provider value={{ looks, setLook }}>{children}</LookContext.Provider>
}

export function useLook() {
  return useContext(LookContext)
}

export function tintHex(hex: string, look: Look) {
  const rgb = hexToRgb(hex)
  if (!rgb) return hex
  let [r, g, b] = rgb
  const temp = (look.temp ?? 0) / 100
  const sat = (look.sat ?? 0) / 100
  const bright = (look.bright ?? 0) / 100
  const amount = (look.amount ?? 0) / 100
  r += temp * 48 + amount * 18 + bright * 36
  g += bright * 28
  b += -temp * 48 + amount * 10 + bright * 24
  const gray = (r + g + b) / 3
  const scale = Math.max(0, 1 + sat)
  r = gray + (r - gray) * scale
  g = gray + (g - gray) * scale
  b = gray + (b - gray) * scale
  return rgbToHex(clamp(r), clamp(g), clamp(b))
}

function clamp(value: number) {
  return Math.max(0, Math.min(255, Math.round(value)))
}

function hexToRgb(hex: string) {
  const match = hex.replace("#", "").match(/^([0-9a-f]{6})$/i)
  if (!match) return null
  const value = match[1]
  return [parseInt(value.slice(0, 2), 16), parseInt(value.slice(2, 4), 16), parseInt(value.slice(4, 6), 16)]
}

function rgbToHex(r: number, g: number, b: number) {
  return `#${[r, g, b].map((channel) => channel.toString(16).padStart(2, "0")).join("")}`
}
