import SwiftUI

/// The editor screen: top chrome, preview, transport, timeline, and a bottom
/// area that swaps between toolbars and property panels.
struct EditorView: View {
    var state: EditorState
    var onHome: () -> Void
    var onAddMedia: () -> Void
    var onAddOverlay: () -> Void
    var onReplaceMedia: () -> Void

    @State private var activePanel: EditorPanel?
    @State private var exportPresented = false
    @State private var fullscreenPreview = false

    var body: some View {
        VStack(spacing: 0) {
            if !fullscreenPreview {
                EditorTopBar(
                    exportEnabled: !state.isEmpty,
                    onHome: onHome,
                    onFullscreen: { toggleFullscreen() },
                    onExport: { exportPresented = true }
                )
            }

            PreviewCanvas(
                state: state,
                onEditText: { id in openPanel(.text(editing: id, tab: 0)) }
            )
            .frame(maxHeight: .infinity)
            .overlay(alignment: .topTrailing) {
                if fullscreenPreview {
                    fullscreenChrome
                }
            }

            if !fullscreenPreview {
                TransportControls(
                    state: state,
                    canUndo: state.canUndo,
                    canRedo: state.canRedo,
                    onSplit: { state.splitAtPlayhead() },
                    onUndo: { state.undo() },
                    onRedo: { state.redo() }
                )

                TimelineView(
                    state: state,
                    onAddMedia: onAddMedia,
                    onTransitionTap: { id in openPanel(.transition(after: id)) }
                )

                bottomArea
            }
        }
        .background(Theme.background)
        .onChange(of: state.selection) {
            if state.selection == nil, let panel = activePanel, panel.requiresSelection {
                closePanel(apply: true)
            }
        }
        .sheet(isPresented: $exportPresented) {
            ExportSheet(duration: state.duration)
        }
    }

    // MARK: Fullscreen preview

    private func toggleFullscreen() {
        withAnimation(.snappy(duration: 0.25)) {
            fullscreenPreview.toggle()
        }
    }

    /// Minimal floating controls shown while the editor chrome is hidden.
    private var fullscreenChrome: some View {
        HStack(spacing: 14) {
            Button {
                state.isPlaying.toggle()
            } label: {
                Image(systemName: state.isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 36, height: 36)
                    .background(.black.opacity(0.55), in: Circle())
            }
            .buttonStyle(.plain)

            Button {
                toggleFullscreen()
            } label: {
                Image(systemName: "arrow.down.right.and.arrow.up.left")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 36, height: 36)
                    .background(.black.opacity(0.55), in: Circle())
            }
            .buttonStyle(.plain)
        }
        .padding(14)
    }

    // MARK: Bottom area (toolbars <-> panels)

    @ViewBuilder
    private var bottomArea: some View {
        if let panel = activePanel {
            PanelSheet(
                title: panel.title,
                showsCancel: panelShowsCancel(panel),
                onCancel: { closePanel(apply: false) },
                onApply: { closePanel(apply: true) }
            ) {
                panelContent(panel)
            }
            .transition(.move(edge: .bottom).combined(with: .opacity))
        } else {
            switch state.selection {
            case .main:
                ClipToolbar(onAdd: onAddMedia, actions: mainClipActions)
            case .overlay(let id):
                LaneToolbar(actions: overlayActions(for: id))
            case .effect:
                LaneToolbar(actions: commonLaneActions)
            case .audio:
                LaneToolbar(actions: audioActions)
            case nil:
                MediaToolbar(
                    onAddMedia: onAddMedia,
                    onAddOverlay: onAddOverlay,
                    onOpenPanel: { openPanel($0) }
                )
            }
        }
    }

    // MARK: Per-kind lane toolbar actions

    /// The full CapCut-style operation strip for a selected main clip.
    private var mainClipActions: [ToolbarAction] {
        [
            ToolbarAction(symbol: "scissors", label: "Split") { state.splitAtPlayhead() },
            ToolbarAction(symbol: "speedometer", label: "Speed") { openPanel(.clipSpeed) },
            ToolbarAction(symbol: "speaker.wave.2", label: "Volume") { openPanel(.clipVolume) },
            ToolbarAction(symbol: "sparkles.rectangle.stack", label: "Animation") { openPanel(.clipAnimation) },
            ToolbarAction(symbol: "camera.filters", label: "Filters") { openPanel(.clipFilter) },
            ToolbarAction(symbol: "slider.horizontal.3", label: "Adjust") { openPanel(.clipAdjust) },
            ToolbarAction(symbol: "circle.righthalf.filled", label: "Opacity") { openPanel(.clipOpacity) },
            ToolbarAction(symbol: "crop", label: "Crop") { openPanel(.clipCrop) },
            ToolbarAction(symbol: "circle.dashed", label: "Mask") { openPanel(.clipMask) },
            ToolbarAction(symbol: "drop", label: "Chroma") { openPanel(.clipChroma) },
            ToolbarAction(symbol: "gyroscope", label: "Stabilize") { openPanel(.clipStabilize) },
            ToolbarAction(symbol: "arrow.uturn.backward.circle", label: "Reverse") { state.reverseSelected() },
            ToolbarAction(symbol: "snowflake", label: "Freeze") { state.freezeFrame() },
            ToolbarAction(symbol: "waveform.badge.plus", label: "Extract\naudio") { state.extractAudio() },
            ToolbarAction(symbol: "plus.square.on.square", label: "Duplicate") { state.duplicateSelected() },
            ToolbarAction(symbol: "rectangle.2.swap", label: "Replace", action: onReplaceMedia),
            ToolbarAction(symbol: "trash", label: "Delete") { state.deleteSelected() },
        ]
    }

    /// Split / duplicate / delete, shared by every lane kind.
    private var commonLaneActions: [ToolbarAction] {
        [
            ToolbarAction(symbol: "scissors", label: "Split") { state.splitAtPlayhead() },
            ToolbarAction(symbol: "plus.square.on.square", label: "Duplicate") { state.duplicateSelected() },
            ToolbarAction(symbol: "trash", label: "Delete") { state.deleteSelected() },
        ]
    }

    private func overlayActions(for id: UUID) -> [ToolbarAction] {
        guard let overlay = state.overlayClips.first(where: { $0.id == id }) else { return [] }
        switch overlay.kind {
        case .text:
            return [
                ToolbarAction(symbol: "pencil", label: "Edit") { openPanel(.text(editing: id, tab: 0)) },
                ToolbarAction(symbol: "textformat", label: "Font") { openPanel(.text(editing: id, tab: 1)) },
                ToolbarAction(symbol: "paintbrush", label: "Style") { openPanel(.text(editing: id, tab: 2)) },
                ToolbarAction(symbol: "sparkles.rectangle.stack", label: "Animation") { openPanel(.text(editing: id, tab: 3)) },
            ] + commonLaneActions
        case .sticker:
            return commonLaneActions
        case .pip:
            return [
                ToolbarAction(symbol: "rectangle.2.swap", label: "Replace", action: onReplaceMedia),
                ToolbarAction(symbol: "speaker.wave.2", label: "Volume") { openPanel(.overlayVolume) },
            ] + commonLaneActions
        }
    }

    private var audioActions: [ToolbarAction] {
        [
            ToolbarAction(symbol: "speaker.wave.2", label: "Volume") { openPanel(.audioVolume) },
            ToolbarAction(symbol: "point.bottomleft.forward.to.point.topright.scurvepath", label: "Fade") { openPanel(.audioFade) },
        ] + commonLaneActions
    }

    func openPanel(_ panel: EditorPanel) {
        if activePanel != nil {
            state.commitPanelSession()
        }
        state.beginPanelSession()
        withAnimation(.easeOut(duration: 0.18)) {
            activePanel = panel
        }
    }

    private func closePanel(apply: Bool) {
        if apply {
            state.commitPanelSession()
        } else {
            state.cancelPanelSession()
        }
        withAnimation(.easeOut(duration: 0.15)) {
            activePanel = nil
        }
    }

    /// Picker-style panels add content instantly; hiding X avoids implying
    /// their additions can be reverted from the header.
    private func panelShowsCancel(_ panel: EditorPanel) -> Bool {
        switch panel {
        case .stickers, .effects, .audio, .captions:
            return false
        default:
            return true
        }
    }

    @ViewBuilder
    private func panelContent(_ panel: EditorPanel) -> some View {
        switch panel {
        case .aspect:
            AspectPanel(state: state)
        case .background:
            BackgroundPanel(state: state)
        case .text(let editing, let tab):
            TextPanel(state: state, editingID: editing, initialTab: tab)
        case .stickers:
            StickersPanel(state: state)
        case .effects:
            EffectsPanel(state: state)
        case .filters:
            FiltersPanel(state: state)
        case .adjust:
            AdjustPanel(state: state)
        case .audio:
            AudioPanel(state: state)
        case .captions:
            CaptionsPanel(state: state, onGenerated: { closePanel(apply: true) })
        case .overlayVolume:
            OverlayVolumePanel(state: state)
        case .audioVolume:
            AudioVolumePanel(state: state)
        case .audioFade:
            AudioFadePanel(state: state)
        case .clipVolume:
            ClipVolumePanel(state: state)
        case .clipSpeed:
            SpeedPanel(state: state)
        case .clipAnimation:
            ClipAnimationPanel(state: state)
        case .clipFilter:
            ClipFilterPanel(state: state)
        case .clipAdjust:
            ClipAdjustPanel(state: state)
        case .clipOpacity:
            OpacityPanel(state: state)
        case .clipCrop:
            CropPanel(state: state)
        case .clipMask:
            MaskPanel(state: state)
        case .clipChroma:
            ChromaPanel(state: state)
        case .clipStabilize:
            StabilizePanel(state: state)
        case .transition(let afterID):
            TransitionPanel(state: state, afterClipID: afterID)
        }
    }
}

#Preview {
    let state = EditorState()
    let _ = state.startProject(with: Array(MockData.libraryItems.prefix(3)))
    return EditorView(state: state, onHome: {}, onAddMedia: {}, onAddOverlay: {}, onReplaceMedia: {})
}
