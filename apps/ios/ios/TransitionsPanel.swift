import SwiftUI

/// Transitions on one cut between main clips: a None tile, then every
/// transition looping its preview under its category's tab. A kept
/// transition shows its length below, and can go on every cut.
struct TransitionsPanel: View {
    let model: EditorModel
    let close: () -> Void
    @State private var category: String
    @State private var seconds: Double

    init(model: EditorModel, close: @escaping () -> Void) {
        self.model = model
        self.close = close
        let kept = model.cut?.transition
        let category = kept.flatMap { kept in
            model.transitionList.first { $0.id == kept }?.category
        } ?? model.transitionTabs.first ?? ""
        _category = State(initialValue: category)
        _seconds = State(initialValue: Double(model.cut?.durationUs ?? 0) / 1_000_000)
    }

    private var cut: Cut? { model.cut }
    private var kept: Bool { cut.map { !$0.name.isEmpty } ?? false }

    var body: some View {
        PanelFrame(title: "Transitions", close: close) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 18) {
                    ForEach(model.transitionTabs, id: \.self) { name in
                        Button { category = name } label: {
                            Text(name.capitalized)
                                .font(.footnote.weight(category == name ? .semibold : .regular))
                                .foregroundStyle(category == name ? .white : .white.opacity(0.5))
                        }
                    }
                }
                .padding(.horizontal, 2)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 10) {
                    noneTile
                    ForEach(model.transitionList.filter { $0.category == category }, id: \.id) {
                        tile($0)
                    }
                }
                .padding(.horizontal, 2)
            }
            .frame(height: 84)
            .id(category)
            if let cut, kept, cut.longestUs > cut.shortestUs {
                HStack(spacing: 12) {
                    Slider(
                        value: $seconds,
                        in: Double(cut.shortestUs) / 1_000_000...Double(cut.longestUs) / 1_000_000
                    ) { editing in
                        if !editing { model.setTransitionLength(Int64(seconds * 1_000_000)) }
                    }
                    .tint(Palette.accent)
                    .accessibilityLabel("Length")
                    .accessibilityValue(String(format: "%.1f seconds", seconds))
                    Text(String(format: "%.1fs", seconds))
                        .font(.footnote.monospacedDigit())
                        .frame(width: 40, alignment: .trailing)
                    Button("Apply to all") { model.transitionEverywhere() }
                        .font(.footnote.weight(.semibold))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color(white: 0.2), in: Capsule())
                }
            }
        }
        .onChange(of: cut?.durationUs) { _, value in
            if let value { seconds = Double(value) / 1_000_000 }
        }
    }

    private var noneTile: some View {
        Button {
            model.setTransition(nil)
        } label: {
            VStack(spacing: 5) {
                HugeIcon("close", size: 22)
                    .frame(width: 60, height: 60)
                    .background(Color(white: 0.18), in: RoundedRectangle(cornerRadius: 8))
                    .overlay {
                        RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(kept ? .clear : Palette.accent, lineWidth: 2)
                    }
                Text("None").font(.caption2)
            }
        }
        .disabled(!kept)
        .accessibilityLabel("No transition")
        .accessibilityAddTraits(kept ? [] : .isSelected)
    }

    private func tile(_ transition: TransitionChoice) -> some View {
        let chosen = cut?.transition == transition.id
        return Button {
            model.setTransition(transition.id)
        } label: {
            VStack(spacing: 5) {
                LoopingVideo(name: transition.preview)
                    .frame(width: 60, height: 60)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay {
                        RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(chosen ? Palette.accent : .clear, lineWidth: 2)
                    }
                Text(transition.name).font(.caption2).lineLimit(1)
                    .frame(width: 64)
            }
        }
        .accessibilityLabel(transition.name)
        .accessibilityAddTraits(chosen ? .isSelected : [])
    }
}
