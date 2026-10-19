const stroke = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true as const,
}

export function Blade() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M3 13 L8 2 H10 L5.2 13 Z" fill="#e4ff57" />
    </svg>
  )
}

export function Icon({
  name,
}: {
  name:
    | "media"
    | "audio"
    | "text"
    | "stickers"
    | "effects"
    | "transitions"
    | "templates"
    | "play"
    | "pause"
    | "back"
    | "forward"
}) {
  if (name === "media") {
    return (
      <svg {...stroke}>
        <rect x="3" y="5" width="18" height="14" rx="2" />
        <path d="M3 15.5 8 11l4 3 3-2 6 4" />
      </svg>
    )
  }
  if (name === "audio") {
    return (
      <svg {...stroke}>
        <path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2" />
      </svg>
    )
  }
  if (name === "text") {
    return (
      <svg {...stroke}>
        <path d="M5 6h14M12 6v13" />
      </svg>
    )
  }
  if (name === "stickers") {
    return (
      <svg {...stroke}>
        <circle cx="12" cy="12" r="8" />
        <path d="M9 10h.01M15 10h.01M8.5 14c1 1.2 2.2 1.8 3.5 1.8S14.5 15.2 15.5 14" />
      </svg>
    )
  }
  if (name === "effects") {
    return (
      <svg {...stroke}>
        <path d="M12 3.5 13.4 8.6 18.5 10 13.4 11.4 12 16.5 10.6 11.4 5.5 10 10.6 8.6 12 3.5Z" />
        <path d="M18 15.5 18.6 17.4 20.5 18 18.6 18.6 18 20.5 17.4 18.6 15.5 18 17.4 17.4 18 15.5Z" />
      </svg>
    )
  }
  if (name === "transitions") {
    return (
      <svg {...stroke}>
        <rect x="3" y="6" width="8" height="12" rx="1.5" />
        <rect x="13" y="6" width="8" height="12" rx="1.5" />
      </svg>
    )
  }
  if (name === "templates") {
    return (
      <svg {...stroke}>
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <path d="M3 9h18M9 9v11" />
      </svg>
    )
  }
  if (name === "play") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M8 6.5v11l10-5.5-10-5.5Z" fill="currentColor" />
      </svg>
    )
  }
  if (name === "pause") {
    return (
      <svg {...stroke}>
        <path d="M9 7v10M15 7v10" />
      </svg>
    )
  }
  if (name === "back") {
    return (
      <svg {...stroke}>
        <path d="M11 7 6 12l5 5M18 7l-5 5 5 5" />
      </svg>
    )
  }
  return (
    <svg {...stroke}>
      <path d="M13 7l5 5-5 5M6 7l5 5-5 5" />
    </svg>
  )
}
