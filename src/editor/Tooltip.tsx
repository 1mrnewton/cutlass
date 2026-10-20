import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react"

export function Tip({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span className="tip-host" data-tip={label}>
      {children}
    </span>
  )
}

type TipBox = { text: string; rect: DOMRect }

function tipOf(node: EventTarget | null) {
  if (!(node instanceof Element)) return null
  const marked = node.closest("[data-tip]") as HTMLElement | null
  const markedText = marked?.dataset.tip?.trim()
  if (marked && markedText) return { el: marked, text: markedText }
  const button = node.closest("button[aria-label]") as HTMLButtonElement | null
  if (!button) return null
  const label = button.getAttribute("aria-label")?.trim()
  if (!label) return null
  const visible = button.innerText.replace(/\s+/g, " ").trim()
  if (visible && visible.toLowerCase() === label.toLowerCase()) return null
  return { el: button, text: label }
}

export function TooltipLayer() {
  const [tip, setTip] = useState<TipBox | null>(null)
  const nodeRef = useRef<HTMLDivElement>(null)
  const pending = useRef<HTMLElement | null>(null)
  const shown = useRef<HTMLElement | null>(null)
  const timer = useRef(0)

  useEffect(() => {
    const root = document.querySelector(".desktop")
    if (!root) return
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches

    const hide = () => {
      window.clearTimeout(timer.current)
      pending.current = null
      shown.current = null
      setTip(null)
    }

    const arm = (found: { el: HTMLElement; text: string }, delay: number) => {
      if (found.el === shown.current || found.el === pending.current) return
      window.clearTimeout(timer.current)
      pending.current = found.el
      timer.current = window.setTimeout(() => {
        if (pending.current !== found.el) return
        shown.current = found.el
        pending.current = null
        setTip({ text: found.text, rect: found.el.getBoundingClientRect() })
      }, delay)
    }

    const over = (event: Event) => {
      const found = tipOf(event.target)
      if (!found || !root.contains(found.el)) {
        const next = event.target
        const current = pending.current ?? shown.current
        if (next instanceof Node && current?.contains(next)) return
        hide()
        return
      }
      arm(found, event.type === "focusin" || reduce ? 0 : 320)
    }

    const out = (event: PointerEvent) => {
      const current = pending.current ?? shown.current
      if (!current) return
      const next = event.relatedTarget
      if (next instanceof Node && current.contains(next)) return
      hide()
    }

    root.addEventListener("pointerover", over)
    root.addEventListener("pointerout", out)
    root.addEventListener("focusin", over)
    root.addEventListener("focusout", hide)
    root.addEventListener("pointerdown", hide)
    root.addEventListener("scroll", hide, true)
    return () => {
      window.clearTimeout(timer.current)
      root.removeEventListener("pointerover", over)
      root.removeEventListener("pointerout", out)
      root.removeEventListener("focusin", over)
      root.removeEventListener("focusout", hide)
      root.removeEventListener("pointerdown", hide)
      root.removeEventListener("scroll", hide, true)
    }
  }, [])

  useLayoutEffect(() => {
    const node = nodeRef.current
    if (!node || !tip) return
    const box = node.getBoundingClientRect()
    let left = tip.rect.left + tip.rect.width / 2 - box.width / 2
    let top = tip.rect.top - box.height - 6
    if (top < 8) top = tip.rect.bottom + 6
    left = Math.max(8, Math.min(left, window.innerWidth - box.width - 8))
    if (top + box.height > window.innerHeight - 8) top = Math.max(8, window.innerHeight - box.height - 8)
    node.style.left = `${left}px`
    node.style.top = `${top}px`
  }, [tip])

  if (!tip) return null
  return (
    <div className="tip" role="tooltip" ref={nodeRef}>
      {tip.text}
    </div>
  )
}
