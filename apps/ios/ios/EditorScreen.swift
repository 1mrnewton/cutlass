import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// The editor: preview on top, the transport, the timeline, then the tools.
struct EditorScreen: View {
    let name: String
    let home: () -> Void
    @State private var model: EditorModel
    @State private var picked: [PhotosPickerItem] = []
    @State private var panel: Panel?
    /// The bar shown while nothing is selected.
    @State private var openBar: ToolBar = .root
    @State private var exporting = false
    /// The overlay picker is open, and what it picked.
    @State private var pickingOverlay = false
    @State private var overlayPicked: [PhotosPickerItem] = []
    /// The Files picker for a sound is open.
    @State private var pickingSound = false
    /// The Files picker for captions is open.
    @State private var pickingCaptions = false
    /// The picker for what replaces the selected clip is open.
    @State private var pickingReplace = false
    @State private var replacePicked: [PhotosPickerItem] = []
    @State private var pickingReplaceSound = false
    /// The picker for a video whose sound comes in is open.
    @State private var pickingExtract = false
    @State private var extractPicked: [PhotosPickerItem] = []

    /// The tool panel open over the tool bar.
    enum Panel: Equatable {
        case volume, speed, effects, ratio, background, graphs, animations
        case crop, opacity, fade
        case filters, adjust, clipEffects
        case stickers, draw, transitions, shapeStyle
        /// Add text, or change the text clip with this id.
        case text(String?)
    }

    init(project: ProjectSummary, home: @escaping () -> Void) {
        name = project.name
        self.home = home
        _model = State(initialValue: EditorModel(project: project.path))
    }

    var body: some View {
        VStack(spacing: 0) {
            TitleBar(title: name, background: Color(white: 0.06)) {
                Button {
                    model.pause()
                    home()
                } label: {
                    HugeIcon("home")
                        .foregroundStyle(Palette.text)
                        .frame(width: 44, height: 44)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Home")
            }
            .overlay(alignment: .trailing) { exportButton }
            preview
            transport
            TimelineStrip(model: model, picked: $picked) { _ in panel = .transitions }
                .frame(maxHeight: .infinity)
                .background(Color(white: 0.09))
            bottom
        }
        .animation(.easeOut(duration: 0.2), value: panel)
        .animation(.easeOut(duration: 0.15), value: model.selectedClip == nil)
        .animation(.easeOut(duration: 0.15), value: openBar)
        .background(Color(white: 0.06))
        .overlay(alignment: .top) { notice }
        .onChange(of: picked) { _, items in importPicked(items) }
        .onChange(of: overlayPicked) { _, items in importOverlays(items) }
        .onChange(of: replacePicked) { _, items in importOne(items) { model.replaceSelected(with: $0) } }
        .onChange(of: extractPicked) { _, items in importOne(items) { model.extractSound(at: $0) } }
        .photosPicker(
            isPresented: $pickingOverlay, selection: $overlayPicked,
            matching: .any(of: [.images, .videos]), preferredItemEncoding: .current)
        .photosPicker(
            isPresented: $pickingReplace, selection: $replacePicked, maxSelectionCount: 1,
            matching: .any(of: [.images, .videos]), preferredItemEncoding: .current)
        .photosPicker(
            isPresented: $pickingExtract, selection: $extractPicked, maxSelectionCount: 1,
            matching: .videos, preferredItemEncoding: .current)
        .fileImporter(
            isPresented: $pickingSound, allowedContentTypes: [.audio],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case let .success(urls): if let url = urls.first { model.importSound(at: url) }
            case let .failure(error): model.show(error)
            }
        }
        .background {
            // One file importer per view, so the others hang off their own.
            Color.clear
                .fileImporter(
                    isPresented: $pickingCaptions, allowedContentTypes: EditorScreen.subtitleTypes,
                    allowsMultipleSelection: false
                ) { result in
                    switch result {
                    case let .success(urls): if let url = urls.first { model.importCaptions(at: url) }
                    case let .failure(error): model.show(error)
                    }
                }
            Color.clear
                .fileImporter(
                    isPresented: $pickingReplaceSound, allowedContentTypes: [.audio],
                    allowsMultipleSelection: false
                ) { result in
                    switch result {
                    case let .success(urls): if let url = urls.first { model.replaceSelected(with: url) }
                    case let .failure(error): model.show(error)
                    }
                }
        }
        .onChange(of: model.selectedClip) { _, id in
            let stays: [Panel] = [
                .text(nil), .effects, .ratio, .background, .filters, .stickers, .draw, .transitions,
            ]
            if id == nil, let open = panel, !stays.contains(open) { panel = nil }
            if id != nil, panel == .transitions { panel = nil }
            if panel == .shapeStyle, id.flatMap(model.shapeStyle(of:)) == nil { panel = nil }
        }
        .onChange(of: panel) { _, open in
            if open != .transitions { model.selectedCut = nil }
        }
        .onDisappear { model.pause() }
        .sheet(isPresented: $exporting) {
            ExportSheet(model: model, name: name)
        }
    }

    /// Export, top right as in the reference phone editor.
    private var exportButton: some View {
        Button {
            model.pause()
            exporting = true
        } label: {
            Text("Export")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Palette.exportInk)
                .padding(.horizontal, 12)
                .frame(height: 30)
                .background(Palette.export, in: Capsule())
                .opacity(model.hasClips ? 1 : 0.4)
        }
        .disabled(!model.hasClips || model.isOpening)
        .padding(.trailing, 12)
    }

    private var preview: some View {
        ZStack {
            // Lighter than the canvas, so the canvas shows its ratio.
            Color(white: 0.06)
            Color.black
                .aspectRatio(model.canvasAspect, contentMode: .fit)
            if let picture = model.picture {
                Image(decorative: picture, scale: 1)
                    .resizable()
                    .aspectRatio(model.canvasAspect, contentMode: .fit)
            }
            if panel == .draw {
                DrawOverlay(model: model)
            } else if !model.isPlaying {
                StageOverlay(model: model) { panel = .text($0) }
            }
            if model.isOpening || model.isImporting {
                ProgressView().tint(.white)
            }
        }
        .frame(maxWidth: .infinity)
        .aspectRatio(16 / 10, contentMode: .fit)
        // A box larger than the preview stays inside it.
        .clipped()
    }

    /// The clock on the left, play pinned to the middle so it never moves
    /// with the clock, undo and redo on the right. A selected picture or
    /// title adds the keyframe button before undo, and the Graphs button
    /// when the playhead is between two of its keyframes.
    private var transport: some View {
        HStack {
            Text(clock(timeUs: model.playheadUs) + " / " + clock(timeUs: model.durationUs))
                .font(.footnote.monospacedDigit())
                .foregroundStyle(.white.opacity(0.7))
            Spacer()
            if let keys = model.keys, keys.inSpan {
                Button { panel = .graphs } label: { HugeIcon("graphs") }
                    .accessibilityLabel("Graphs")
                    .padding(.trailing, 16)
            }
            if let keys = model.keys {
                Button { model.toggleKeyframe() } label: {
                    HugeIcon("keyframe")
                        .overlay(alignment: .topTrailing) {
                            Text(keys.here ? "−" : "+")
                                .font(.system(size: 11, weight: .bold))
                                .offset(x: 5, y: -5)
                        }
                }
                .accessibilityLabel(keys.here ? "Remove keyframe" : "Add keyframe")
                .padding(.trailing, 16)
            }
            if model.keys == nil {
                Button { model.toggleLinkage() } label: {
                    HugeIcon(model.linkage ? "linkage" : "unlink")
                        .foregroundStyle(model.linkage ? Palette.accent : .white)
                }
                .accessibilityLabel("Linkage")
                .accessibilityValue(model.linkage ? "On" : "Off")
                .padding(.trailing, 16)
            }
            Button { model.undo() } label: { HugeIcon("undo") }
                .accessibilityLabel("Undo")
            Button { model.redo() } label: { HugeIcon("redo") }
                .accessibilityLabel("Redo")
                .padding(.leading, 16)
        }
        .overlay {
            Button {
                model.togglePlay()
            } label: {
                HugeIcon(model.isPlaying ? "pause" : "play", size: 24)
            }
            .disabled(!model.hasClips)
            .accessibilityLabel(model.isPlaying ? "Pause" : "Play")
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    /// A panel when one is open, the selected clip's bar, else the bar the
    /// root tools opened. Rust lists every bar.
    @ViewBuilder
    private var bottom: some View {
        if panel == .effects {
            EffectsPanel(model: model) { self.panel = nil }
                .transition(.move(edge: .bottom))
        } else if panel == .ratio {
            RatioPanel(model: model) { self.panel = nil }
                .transition(.move(edge: .bottom))
        } else if panel == .background {
            BackgroundPanel(model: model) { self.panel = nil }
                .transition(.move(edge: .bottom))
        } else if panel == .animations {
            AnimationsPanel(model: model) { self.panel = nil }
                .transition(.move(edge: .bottom))
        } else if panel == .graphs {
            GraphsPanel(model: model) { self.panel = nil }
                .transition(.move(edge: .bottom))
        } else if panel == .filters {
            FiltersPanel(model: model) { self.panel = nil }
                .transition(.move(edge: .bottom))
        } else if panel == .adjust, model.adjustSliders != nil {
            AdjustPanel(model: model) { self.panel = nil }
                .id(model.selectedClip)
                .transition(.move(edge: .bottom))
        } else if panel == .clipEffects, let clip = model.selectedClip {
            EffectsPanel(model: model, title: "Effects", pick: { model.addClipEffect($0) }) {
                self.panel = nil
            }
            .id(clip)
            .transition(.move(edge: .bottom))
        } else if panel == .stickers {
            StickersPanel(model: model) { self.panel = nil }
                .transition(.move(edge: .bottom))
        } else if panel == .draw {
            DrawPanel(model: model) { self.panel = nil }
                .transition(.move(edge: .bottom))
        } else if panel == .shapeStyle, let clip = model.selectedClip,
                  model.shapeStyle(of: clip) != nil {
            ShapeStylePanel(model: model, clip: clip) { self.panel = nil }
                .id(clip)
                .transition(.move(edge: .bottom))
        } else if panel == .transitions, model.cut != nil {
            TransitionsPanel(model: model) { self.panel = nil }
                .transition(.move(edge: .bottom))
        } else if case let .text(clip) = panel {
            TextPanel(model: model, clip: clip) { self.panel = nil }
                .id(clip ?? "new")
                .transition(.move(edge: .bottom))
        } else if let panel, let clip = model.selection {
            Group {
                switch panel {
                case .volume: VolumePanel(model: model, clip: clip) { self.panel = nil }
                case .speed: SpeedPanel(model: model, clip: clip) { self.panel = nil }
                case .crop: CropPanel(model: model) { self.panel = nil }
                case .opacity: OpacityPanel(model: model) { self.panel = nil }
                case .fade: FadePanel(model: model, clip: clip) { self.panel = nil }
                case .text, .effects, .ratio, .background, .graphs, .animations,
                     .filters, .adjust, .clipEffects, .stickers, .draw, .transitions, .shapeStyle:
                    EmptyView()
                }
            }
            .transition(.move(edge: .bottom))
        } else if let bar = model.selectionBar {
            ToolBarView(tools: toolBar(bar: bar), back: { model.selectedClip = nil }, tap: use)
                .id(bar)
        } else {
            ToolBarView(
                tools: toolBar(bar: openBar),
                back: openBar == .root ? nil : { openBar = .root },
                tap: use
            )
            .id(openBar)
        }
    }

    /// Open a bar, act, or say the tool is not ready yet.
    private func use(_ tool: ToolItem) {
        if let bar = tool.opens {
            openBar = bar
            return
        }
        guard tool.ready else {
            model.say("\(tool.label) is coming soon")
            return
        }
        switch tool.kind {
        case .edit: model.editUnderPlayhead()
        case .split: model.split()
        case .volume: panel = .volume
        case .speed: panel = .speed
        case .animations: panel = .animations
        case .delete: model.deleteClip()
        case .addText: panel = .text(nil)
        case .editText: panel = .text(model.selectedClip)
        case .videoEffects: panel = .effects
        case .aspectRatio:
            model.selectedClip = nil
            panel = .ratio
        case .background:
            model.selectedClip = nil
            panel = .background
        case .duplicate: model.duplicate()
        case .crop: panel = .crop
        case .mirror: model.toggleMirror()
        case .rotate: model.turn()
        case .opacity: panel = .opacity
        case .fade: panel = .fade
        case .reverse: model.toggleReverse()
        case .freeze: model.freeze()
        case .extractAudio:
            // The Audio bar takes a video's sound; a clip gives up its own.
            if model.selectedClip == nil { pickingExtract = true } else { model.extractOrRecoverAudio() }
        case .beats: model.toggleBeat()
        case .splice: model.toMain()
        case .addOverlay: pickingOverlay = true
        case .importSound: pickingSound = true
        case .filters: panel = .filters
        case .adjust:
            if model.selectedClip == nil { model.addAdjustment() }
            panel = .adjust
        case .effects: panel = .clipEffects
        case .stickers: panel = .stickers
        case .draw:
            model.pause()
            model.selectedClip = nil
            panel = .draw
        case .captions: pickingCaptions = true
        case .shapeStyle: panel = .shapeStyle
        case .replace:
            if model.selectionBar == .audioClip { pickingReplaceSound = true } else { pickingReplace = true }
        default: model.say("\(tool.label) is coming soon")
        }
    }

    @ViewBuilder
    private var notice: some View {
        if let notice = model.notice {
            Text(notice.text)
                .font(.footnote.weight(.medium))
                .lineLimit(2)
                .foregroundStyle(.black)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.white, in: Capsule())
                .padding(.top, 8)
                .transition(.move(edge: .top).combined(with: .opacity))
                .task(id: notice.id) {
                    try? await Task.sleep(for: .seconds(2.5))
                    if model.notice == notice { withAnimation { model.clearNotice() } }
                }
        }
    }

    /// SRT and LRC files, by extension, or any text.
    static let subtitleTypes: [UTType] =
        [UTType(filenameExtension: "srt"), UTType(filenameExtension: "lrc")].compactMap { $0 }
        + [.plainText]

    /// Hand the one picked item to `use`, then drop the copy.
    private func importOne(_ items: [PhotosPickerItem], use: @escaping (URL) -> Void) {
        guard let item = items.first else { return }
        replacePicked = []
        extractPicked = []
        model.beginImport()
        Task {
            do {
                let file = try await PickedFile.load(item)
                use(file)
                try? FileManager.default.removeItem(at: file.deletingLastPathComponent())
            } catch {
                model.show(error)
            }
            model.endImport()
        }
    }

    /// Bring each picked item in as an overlay at the playhead.
    private func importOverlays(_ items: [PhotosPickerItem]) {
        guard !items.isEmpty else { return }
        overlayPicked = []
        model.beginImport()
        Task {
            for item in items {
                do {
                    let file = try await PickedFile.load(item)
                    model.addOverlay(at: file)
                    try? FileManager.default.removeItem(at: file.deletingLastPathComponent())
                } catch {
                    model.show(error)
                }
            }
            model.endImport()
        }
    }

    private func importPicked(_ items: [PhotosPickerItem]) {
        guard !items.isEmpty else { return }
        picked = []
        model.beginImport()
        Task {
            for item in items {
                do {
                    let file = try await PickedFile.load(item)
                    model.importFile(at: file)
                    try? FileManager.default.removeItem(at: file.deletingLastPathComponent())
                } catch {
                    model.show(error)
                }
            }
            model.endImport()
        }
    }
}
