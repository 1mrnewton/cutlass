import Photos
import SwiftUI
import os

/// Export the whole cut as an MP4 and save it to Photos.
///
/// Rust draws, encodes and writes the file; this sheet picks the size,
/// shows the frames done, stops it on request, and hands the finished file
/// to Photos or the share sheet. The file is written in the temporary
/// folder and removed when the sheet closes.
struct ExportSheet: View {
    let model: EditorModel
    let name: String
    @Environment(\.dismiss) private var dismiss
    @State private var side: UInt32 = 1080
    @State private var phase: Phase = .choosing
    @State private var listener: ExportListener?

    enum Phase: Equatable {
        case choosing
        /// The share of the frames written, 0 to 1.
        case writing(Double)
        case saving
        /// The file, and whether Photos took it.
        case finished(URL, saved: Bool)
        case failed(String)
    }

    private static let sides = exportSides()

    var body: some View {
        VStack(spacing: 20) {
            header
            switch phase {
            case .choosing: choosing
            case let .writing(share): writing(share)
            case .saving: writing(1)
            case let .finished(file, saved): finished(file, saved: saved)
            case let .failed(message): failed(message)
            }
        }
        .foregroundStyle(Palette.text)
        .padding(.horizontal, 20)
        .padding(.top, 16)
        .padding(.bottom, 24)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(Palette.panel)
        .presentationDetents([.height(280)])
        .presentationDragIndicator(busy ? .hidden : .visible)
        .interactiveDismissDisabled(busy)
        .onDisappear { removeFile() }
    }

    private var busy: Bool {
        switch phase {
        case .writing, .saving: true
        default: false
        }
    }

    private var header: some View {
        ZStack {
            Text("Export").font(.headline)
            if !busy {
                HStack {
                    Spacer()
                    Button { dismiss() } label: {
                        HugeIcon("close", size: 20)
                            .frame(width: 40, height: 32)
                            .contentShape(Rectangle())
                    }
                    .accessibilityLabel("Close")
                }
            }
        }
    }

    private var choosing: some View {
        VStack(spacing: 20) {
            Picker("Resolution", selection: $side) {
                ForEach(Self.sides, id: \.self) { side in
                    Text(Self.label(side)).tag(side)
                }
            }
            .pickerStyle(.segmented)
            Text("H.264 and AAC, saved to Photos")
                .font(.footnote)
                .foregroundStyle(Palette.muted)
            wideButton("Export", action: start)
        }
    }

    private func writing(_ share: Double) -> some View {
        VStack(spacing: 14) {
            ProgressView(value: share)
                .tint(Palette.accent)
            Text(phase == .saving ? "Saving to Photos" : "\(Int(share * 100))%")
                .font(.footnote.monospacedDigit())
                .foregroundStyle(Palette.muted)
            if phase != .saving {
                Button("Stop") { listener?.stop() }
                    .font(.subheadline.weight(.semibold))
                    .padding(.top, 8)
            }
        }
        .padding(.top, 12)
    }

    private func finished(_ file: URL, saved: Bool) -> some View {
        VStack(spacing: 16) {
            HStack(spacing: 8) {
                HugeIcon("saved", size: 20)
                    .foregroundStyle(saved ? Palette.accent : Palette.muted)
                Text(saved ? "Saved to Photos" : "Photos access is off. Share the movie instead.")
                    .font(.subheadline)
                    .multilineTextAlignment(.center)
            }
            ShareLink(item: file) {
                HStack(spacing: 6) {
                    HugeIcon("share", size: 18)
                    Text("Share")
                }
                .font(.subheadline.weight(.semibold))
            }
            wideButton("Done") { dismiss() }
        }
        .padding(.top, 8)
    }

    private func failed(_ message: String) -> some View {
        VStack(spacing: 16) {
            Text(message)
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundStyle(Palette.muted)
            wideButton("Try again") { phase = .choosing }
        }
        .padding(.top, 8)
    }

    private func wideButton(_ title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.headline)
                .foregroundStyle(Palette.exportInk)
                .frame(maxWidth: .infinity)
                .frame(height: 46)
                .background(Palette.export, in: RoundedRectangle(cornerRadius: 10))
        }
    }

    static func label(_ side: UInt32) -> String {
        side >= 2160 ? "4K" : "\(side)p"
    }

    /// Write the movie off the main thread, then save it to Photos.
    private func start() {
        guard let editor = model.beginExport() else {
            dismiss()
            return
        }
        removeFile()
        let file = Self.file(named: name)
        let side = side
        phase = .writing(0)
        let listener = ExportListener { share in
            Task { @MainActor in
                if case .writing = phase { phase = .writing(share) }
            }
        }
        self.listener = listener
        Task {
            let written = await Task.detached(priority: .userInitiated) {
                Result {
                    try editor.export(
                        path: file.path(percentEncoded: false),
                        shortSide: side,
                        progress: listener
                    )
                }
            }.value
            model.endExport()
            self.listener = nil
            switch written {
            case .success:
                phase = .saving
                let saved = await Self.saveToPhotos(file)
                phase = .finished(file, saved: saved)
            case let .failure(error):
                if listener.stopped {
                    phase = .choosing
                } else if case let EditorError.Failed(message) = error {
                    phase = .failed(message)
                } else {
                    phase = .failed(error.localizedDescription)
                }
            }
        }
    }

    private func removeFile() {
        try? FileManager.default.removeItem(at: Self.folder)
    }

    private static let log = Logger(subsystem: "cutlass", category: "export")

    /// Where exports wait to be saved or shared.
    private static var folder: URL {
        FileManager.default.temporaryDirectory.appending(path: "Exports", directoryHint: .isDirectory)
    }

    /// A file in [`folder`] named after the project.
    private static func file(named name: String) -> URL {
        let kept = name.unicodeScalars.filter {
            CharacterSet.alphanumerics.contains($0) || $0 == " " || $0 == "-"
        }
        let trimmed = String(String.UnicodeScalarView(kept)).trimmingCharacters(in: .whitespaces)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder.appending(path: (trimmed.isEmpty ? "Cutlass" : trimmed) + ".mp4")
    }

    /// Add the movie to Photos, asking first. False when the person said no
    /// or Photos refused it.
    private static func saveToPhotos(_ file: URL) async -> Bool {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else { return false }
        do {
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetCreationRequest.forAsset().addResource(with: .video, fileURL: file, options: nil)
            }
            return true
        } catch {
            log.error("photos refused the movie: \(error.localizedDescription, privacy: .public)")
            return false
        }
    }
}

/// Hears Rust's progress on the editor's thread. Reports only whole
/// percents, so the main thread is not woken for every frame.
final class ExportListener: ExportProgress {
    private let report: @Sendable (Double) -> Void
    private let state = OSAllocatedUnfairLock(initialState: (stop: false, percent: -1))

    init(report: @escaping @Sendable (Double) -> Void) {
        self.report = report
    }

    var stopped: Bool { state.withLock { $0.stop } }

    func stop() {
        state.withLock { $0.stop = true }
    }

    func progress(done: UInt64, total: UInt64) -> Bool {
        let share = total == 0 ? 1 : Double(done) / Double(total)
        let percent = Int(share * 100)
        let (stop, fresh) = state.withLock { state in
            let fresh = percent != state.percent
            state.percent = percent
            return (state.stop, fresh)
        }
        if fresh { report(share) }
        return !stop
    }
}
