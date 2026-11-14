import SwiftUI

/// The assistant: the conversation, a box to ask in, and Stop while it
/// works. Until the cloud model is on with a key, it shows its settings
/// instead. Rust runs the model and the tools; this only shows the lines.
struct AssistantPanel: View {
    let model: EditorModel
    let done: () -> Void
    @State private var cloud: CloudChoice?
    @State private var setting = true

    var body: some View {
        PanelFrame(title: "Assistant", close: done) {
            if let current = cloud {
                if setting {
                    CloudSettings(current: current) { choice, key in
                        model.setAssistantCloud(choice, key: key)
                        refresh()
                    }
                } else {
                    Conversation(model: model) { setting = true }
                }
            } else {
                Text("The assistant opens with the project.").foregroundStyle(Palette.text)
            }
        }
        .onAppear(perform: refresh)
    }

    private func refresh() {
        cloud = model.assistantCloud()
        setting = cloud.map { !$0.on || !$0.hasKey } ?? true
    }
}

private struct Conversation: View {
    let model: EditorModel
    let settings: () -> Void
    @State private var ask = ""

    var body: some View {
        ScrollViewReader { reader in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    ForEach(Array(model.assistantLines.enumerated()), id: \.offset) { index, line in
                        LineView(line: line).id(index)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minHeight: 80, maxHeight: 260)
            .onChange(of: model.assistantLines.count) { _, count in
                if count > 0 { withAnimation { reader.scrollTo(count - 1, anchor: .bottom) } }
            }
        }
        HStack(spacing: 8) {
            TextField("Ask for an edit", text: $ask, axis: .vertical)
                .lineLimit(1...4)
                .textFieldStyle(.roundedBorder)
            if model.assistantBusy {
                Pill(label: "Stop") { model.stopAssistant() }
            } else {
                Pill(label: "Send", enabled: !ask.trimmingCharacters(in: .whitespaces).isEmpty) {
                    model.ask(ask)
                    ask = ""
                }
            }
        }
        HStack {
            Button("Clear") { model.clearAssistant() }.disabled(model.assistantBusy)
            Spacer()
            Button("Model and key", action: settings).disabled(model.assistantBusy)
        }
        .foregroundStyle(Palette.text)
        .font(.subheadline)
    }
}

/// One line: what was asked, said, a step and how it went, or a notice.
private struct LineView: View {
    let line: AssistantLine

    var body: some View {
        switch line.kind {
        case .asked:
            Text(line.text)
                .fontWeight(.semibold)
                .foregroundStyle(Palette.text)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
        case .said:
            Text(line.text).font(.subheadline).foregroundStyle(Palette.text)
                .fixedSize(horizontal: false, vertical: true)
        case .step:
            let state = line.running ? "…" : line.failed ? "failed: \(line.result.prefix(160))" : "done"
            Text("\(line.tool)  \(state)")
                .font(.caption)
                .foregroundStyle(line.failed ? Color(red: 1, green: 0.54, blue: 0.5) : Palette.muted)
        case .notice:
            Text(line.text).font(.footnote).foregroundStyle(Color(red: 1, green: 0.82, blue: 0.5))
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// The cloud model's service, name and key. Turning it on sends what is
/// asked off the phone, so it asks first.
private struct CloudSettings: View {
    let current: CloudChoice
    let save: (CloudChoice, String?) -> Void
    @State private var provider = ""
    @State private var name = ""
    @State private var url = ""
    @State private var key = ""
    @State private var asking = false

    private var keptKey: Bool { current.hasKey && provider == current.provider }

    var body: some View {
        Text(
            "The assistant uses a cloud model with your own key. What you ask and the project's outline are sent to that service. The key stays on this phone, apart from the settings."
        )
        .font(.footnote)
        .foregroundStyle(Palette.muted)
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
        HStack(spacing: 8) {
            Choice(label: "Claude", chosen: provider == "claude") {
                if provider != "claude" {
                    provider = "claude"
                    name = "claude-opus-5-5"
                }
            }
            Choice(label: "Chat completions", chosen: provider == "openai") {
                if provider != "openai" {
                    provider = "openai"
                    name = ""
                }
            }
        }
        if provider == "openai" {
            TextField("https://api.openai.com/v1/chat/completions", text: $url)
                .textFieldStyle(.roundedBorder)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
        }
        TextField("Model", text: $name)
            .textFieldStyle(.roundedBorder)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
        SecureField(keptKey ? "Key (kept; type to replace)" : "Key", text: $key)
            .textFieldStyle(.roundedBorder)
        Pill(
            label: "Use the cloud model",
            enabled: (!key.isEmpty || keptKey) && !name.trimmingCharacters(in: .whitespaces).isEmpty
        ) { asking = true }
            .onAppear {
                provider = current.provider
                name = current.name
                url = current.url
            }
            .alert("Send to the cloud?", isPresented: $asking) {
                Button("Allow") {
                    save(
                        CloudChoice(
                            on: true, provider: provider, name: name, url: url,
                            hasKey: !key.isEmpty || keptKey),
                        key.isEmpty ? nil : key)
                }
                Button("Don't allow", role: .cancel) {}
            } message: {
                Text(
                    "What you ask the assistant, and the project's outline it reads, go to \(provider == "claude" ? "Anthropic" : "the chat completions service"). Your media files stay on this phone."
                )
            }
    }
}

private struct Choice: View {
    let label: String
    let chosen: Bool
    let pick: () -> Void

    var body: some View {
        Button(action: pick) {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(chosen ? Palette.exportInk : Palette.text)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(chosen ? Palette.export : Color.white.opacity(0.08), in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

private struct Pill: View {
    let label: String
    var enabled = true
    let tap: () -> Void

    var body: some View {
        Button(action: tap) {
            Text(label)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Palette.exportInk)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Palette.export, in: Capsule())
                .opacity(enabled ? 1 : 0.4)
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
    }
}
