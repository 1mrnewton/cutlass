import { useState } from "react"
import { suggestions } from "./model"

export function CommandBox({ onCommand }: { onCommand: (command: string) => void }) {
  const [draft, setDraft] = useState("")

  const submit = (command: string) => {
    const next = command.trim()
    if (!next) return
    onCommand(next)
    setDraft("")
  }

  return (
    <form
      className="command-bar"
      onSubmit={(event) => {
        event.preventDefault()
        submit(draft)
      }}
    >
      <input
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        placeholder="Remove silences, insert a shot at 0:32, cut a wedding film…"
        aria-label="Edit command"
        autoComplete="off"
      />
      <div className="chips">
        {suggestions.map((item) => (
          <button key={item.label} type="button" className="chip" onClick={() => submit(item.command)}>
            {item.label}
          </button>
        ))}
      </div>
    </form>
  )
}
