import SwiftUI

/// Mock export flow: settings (resolution / frame rate / quality / estimated
/// size) -> fake progress ring -> saved confirmation. No real rendering.
struct ExportSheet: View {
    var duration: TimeInterval

    private enum Phase {
        case settings
        case exporting
        case saved
    }

    private struct Resolution: Hashable {
        var label: String
        var detail: String
        /// Rough H.264 bitrate in megabits per second at quality 1.
        var mbps: Double
    }

    private static let resolutions: [Resolution] = [
        Resolution(label: "720p", detail: "HD", mbps: 7),
        Resolution(label: "1080p", detail: "Full HD", mbps: 14),
        Resolution(label: "4K", detail: "Ultra HD", mbps: 45),
    ]
    private static let frameRates: [Int] = [24, 30, 60]

    @Environment(\.dismiss) private var dismiss
    @State private var phase: Phase = .settings
    @State private var resolution = Self.resolutions[1]
    @State private var frameRate = 30
    @State private var quality: Double = 0.8
    @State private var progress: Double = 0
    @State private var exportTask: Task<Void, Never>?

    var body: some View {
        VStack(spacing: 0) {
            Capsule()
                .fill(Theme.textTertiary.opacity(0.5))
                .frame(width: 36, height: 4)
                .padding(.top, 10)

            switch phase {
            case .settings:
                settings
            case .exporting:
                exporting
            case .saved:
                saved
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Theme.surface)
        .presentationDetents([.medium])
        .interactiveDismissDisabled(phase == .exporting)
        .onDisappear { exportTask?.cancel() }
    }

    // MARK: Settings

    private var settings: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Export")
                .font(.title3.bold())
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity)
                .padding(.top, 14)

            VStack(alignment: .leading, spacing: 10) {
                Text("Resolution")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(Theme.textSecondary)
                HStack(spacing: 10) {
                    ForEach(Self.resolutions, id: \.self) { option in
                        selectablePill(
                            title: option.label,
                            subtitle: option.detail,
                            isOn: option == resolution
                        ) {
                            resolution = option
                        }
                    }
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                Text("Frame rate")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(Theme.textSecondary)
                HStack(spacing: 10) {
                    ForEach(Self.frameRates, id: \.self) { fps in
                        selectablePill(
                            title: "\(fps)",
                            subtitle: "fps",
                            isOn: fps == frameRate
                        ) {
                            frameRate = fps
                        }
                    }
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Quality")
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(Theme.textSecondary)
                    Spacer()
                    Text(qualityLabel)
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(.white)
                }
                Slider(value: $quality, in: 0.2...1)
                    .tint(Theme.accent)
            }

            HStack {
                Label(duration.timecode, systemImage: "clock")
                Spacer()
                Label(estimatedSize, systemImage: "internaldrive")
            }
            .font(.footnote)
            .foregroundStyle(Theme.textSecondary)

            Button {
                startExport()
            } label: {
                Text("Export")
                    .font(.headline)
                    .foregroundStyle(.black)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 13)
                    .background(.white, in: Capsule())
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 22)
    }

    private var qualityLabel: String {
        switch quality {
        case ..<0.45: "Low"
        case ..<0.75: "Medium"
        case ..<0.95: "High"
        default: "Best"
        }
    }

    private var estimatedSize: String {
        let mbps = resolution.mbps * (0.35 + 0.65 * quality) * (Double(frameRate) / 30)
        let megabytes = mbps * max(duration, 0) / 8
        if megabytes >= 1000 {
            return String(format: "~%.1f GB", megabytes / 1000)
        }
        return String(format: "~%.0f MB", max(megabytes, 1))
    }

    // MARK: Exporting

    private func startExport() {
        withAnimation(.snappy(duration: 0.2)) { phase = .exporting }
        progress = 0
        exportTask = Task {
            while !Task.isCancelled, progress < 1 {
                try? await Task.sleep(for: .milliseconds(60))
                withAnimation(.linear(duration: 0.06)) {
                    progress = min(1, progress + Double.random(in: 0.008...0.028))
                }
            }
            guard !Task.isCancelled else { return }
            try? await Task.sleep(for: .milliseconds(250))
            withAnimation(.snappy(duration: 0.25)) { phase = .saved }
        }
    }

    private var exporting: some View {
        VStack(spacing: 22) {
            ZStack {
                Circle()
                    .stroke(Theme.surfaceElevated, lineWidth: 7)
                Circle()
                    .trim(from: 0, to: progress)
                    .stroke(Theme.accent, style: StrokeStyle(lineWidth: 7, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                Text("\(Int(progress * 100))%")
                    .font(.title2.bold().monospacedDigit())
                    .foregroundStyle(.white)
            }
            .frame(width: 116, height: 116)
            .padding(.top, 36)

            VStack(spacing: 5) {
                Text("Exporting...")
                    .font(.headline)
                    .foregroundStyle(.white)
                Text("\(resolution.label) · \(frameRate) fps · \(qualityLabel)")
                    .font(.footnote)
                    .foregroundStyle(Theme.textSecondary)
            }

            Button("Cancel") {
                exportTask?.cancel()
                withAnimation(.snappy(duration: 0.2)) { phase = .settings }
            }
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(Theme.textSecondary)
            .buttonStyle(.plain)
        }
    }

    // MARK: Saved

    private var saved: some View {
        VStack(spacing: 18) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 54))
                .foregroundStyle(Theme.accent)
                .padding(.top, 40)

            VStack(spacing: 5) {
                Text("Saved to Photos")
                    .font(.headline)
                    .foregroundStyle(.white)
                Text("\(resolution.label) · \(frameRate) fps · \(estimatedSize)")
                    .font(.footnote)
                    .foregroundStyle(Theme.textSecondary)
            }

            Button {
                dismiss()
            } label: {
                Text("Done")
                    .font(.headline)
                    .foregroundStyle(.black)
                    .padding(.horizontal, 52)
                    .padding(.vertical, 12)
                    .background(.white, in: Capsule())
            }
            .buttonStyle(.plain)
            .padding(.top, 8)
        }
    }

    // MARK: Pieces

    private func selectablePill(
        title: String,
        subtitle: String,
        isOn: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 2) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(isOn ? .black : .white)
                Text(subtitle)
                    .font(.caption2)
                    .foregroundStyle(isOn ? .black.opacity(0.6) : Theme.textTertiary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 9)
            .background(
                isOn ? AnyShapeStyle(.white) : AnyShapeStyle(Theme.surfaceElevated),
                in: RoundedRectangle(cornerRadius: 11, style: .continuous)
            )
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    Color.black.sheet(isPresented: .constant(true)) {
        ExportSheet(duration: 57)
    }
}
