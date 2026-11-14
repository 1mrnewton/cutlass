import SwiftUI

/// Filters: a None tile, then every filter over its own scene, under its
/// category's tab. A kept filter shows its strength below. With a picture
/// or filter layer selected a tap swaps its filter; with nothing selected
/// it puts a filter layer at the playhead.
struct FiltersPanel: View {
    let model: EditorModel
    let close: () -> Void
    @State private var category: String
    @State private var strength: Double

    init(model: EditorModel, close: @escaping () -> Void) {
        self.model = model
        self.close = close
        let kept = model.clipFilter
        let first = model.filterList.first?.category ?? ""
        let category = kept.flatMap { kept in
            model.filterList.first { $0.id == kept.filterId }?.category
        } ?? first
        _category = State(initialValue: category)
        _strength = State(initialValue: kept?.strength ?? 1)
    }

    private var categories: [String] {
        var seen: [String] = []
        for filter in model.filterList where !seen.contains(filter.category) {
            seen.append(filter.category)
        }
        return seen
    }

    var body: some View {
        PanelFrame(title: "Filters", close: close) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 18) {
                    ForEach(categories, id: \.self) { name in
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
                HStack(spacing: 10) {
                    noneTile
                    ForEach(model.filterList.filter { $0.category == category }, id: \.id) { filter in
                        tile(filter)
                    }
                }
                .padding(.horizontal, 2)
            }
            if model.clipFilter != nil {
                HStack(spacing: 12) {
                    Slider(value: $strength, in: 0...1) { editing in
                        if !editing { model.setFilterStrength(strength) }
                    }
                    .tint(Palette.accent)
                    .accessibilityLabel("Strength")
                    .accessibilityValue("\(Int((strength * 100).rounded()))%")
                    Text("\(Int((strength * 100).rounded()))")
                        .font(.footnote.monospacedDigit())
                        .frame(width: 32, alignment: .trailing)
                }
            }
        }
        .onChange(of: model.clipFilter?.strength) { _, value in
            if let value { strength = value }
        }
    }

    private var noneTile: some View {
        let chosen = model.clipFilter == nil
        return Button {
            model.pickFilter("")
        } label: {
            VStack(spacing: 5) {
                HugeIcon("close", size: 22)
                    .frame(width: 60, height: 60)
                    .background(Color(white: 0.18), in: RoundedRectangle(cornerRadius: 8))
                    .overlay {
                        RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(chosen ? Palette.accent : .clear, lineWidth: 2)
                    }
                Text("None").font(.caption2)
            }
        }
        .accessibilityLabel("No filter")
        .accessibilityAddTraits(chosen ? .isSelected : [])
    }

    private func tile(_ filter: FilterChoice) -> some View {
        let chosen = model.clipFilter?.filterId == filter.id
        return Button {
            model.pickFilter(filter.id)
        } label: {
            VStack(spacing: 5) {
                Group {
                    if let poster = model.poster(filter.id) {
                        Image(decorative: poster, scale: 1).resizable()
                    } else {
                        Color(white: 0.18)
                    }
                }
                .frame(width: 60, height: 60)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay {
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(chosen ? Palette.accent : .clear, lineWidth: 2)
                }
                Text(filter.name).font(.caption2).lineLimit(1)
                    .frame(width: 64)
            }
        }
        .accessibilityLabel(filter.name)
        .accessibilityAddTraits(chosen ? .isSelected : [])
    }
}

/// Adjust: one chip per slider, and the chosen slider below. Each slider
/// is sent when the finger lifts, as one undo step.
struct AdjustPanel: View {
    let model: EditorModel
    let close: () -> Void
    @State private var key = "brightness"
    @State private var value: Double = 0

    private var sliders: [AdjustSlider] { model.adjustSliders ?? [] }
    private var current: AdjustSlider? { sliders.first { $0.key == key } }

    var body: some View {
        PanelFrame(title: "Adjust", close: close) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(sliders, id: \.key) { slider in
                        Button {
                            key = slider.key
                            value = slider.value
                        } label: {
                            VStack(spacing: 4) {
                                Text(slider.value == 0 ? "0" : "\(Int((slider.value * 100).rounded()))")
                                    .font(.caption2.monospacedDigit())
                                    .frame(width: 40, height: 40)
                                    .background(
                                        Circle().fill(slider.key == key ? Palette.accent.opacity(0.25) : Color(white: 0.18)))
                                    .overlay {
                                        Circle().strokeBorder(
                                            slider.key == key ? Palette.accent : .clear, lineWidth: 1.5)
                                    }
                                Text(slider.label).font(.caption2).lineLimit(1)
                            }
                            .frame(width: 64)
                        }
                        .accessibilityLabel(slider.label)
                        .accessibilityAddTraits(slider.key == key ? .isSelected : [])
                    }
                }
                .padding(.horizontal, 2)
            }
            if let current {
                HStack(spacing: 12) {
                    Slider(value: $value, in: current.min...1) { editing in
                        if !editing { model.setAdjust(current.key, value) }
                    }
                    .tint(Palette.accent)
                    .accessibilityLabel(current.label)
                    .accessibilityValue("\(Int((value * 100).rounded()))")
                    Text("\(Int((value * 100).rounded()))")
                        .font(.footnote.monospacedDigit())
                        .frame(width: 36, alignment: .trailing)
                }
            }
        }
        .onAppear { value = current?.value ?? 0 }
        .onChange(of: current?.value) { _, stored in
            if let stored { value = stored }
        }
    }
}
