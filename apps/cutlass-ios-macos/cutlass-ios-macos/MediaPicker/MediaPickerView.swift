import PhotosUI
import SwiftUI

/// Full-screen media picker: the system photo library (inline PhotosPicker)
/// next to the bundled sample files, both committing as file URLs the engine
/// imports. Cancel/Skip pills and the floating action bar frame the system
/// UI so the flow matches the rest of the app.
struct MediaPickerView: View {
    private enum Tab: String, CaseIterable {
        case photos = "Photos"
        case samples = "Samples"
    }

    var onCancel: () -> Void
    var onDone: ([URL]) -> Void

    @State private var tab: Tab = .photos
    @State private var photoPicks: [PhotosPickerItem] = []
    @State private var samplePicks: [URL] = []
    /// True while picks copy out of the photo library.
    @State private var isStaging = false

    private let columns = Array(repeating: GridItem(.flexible(), spacing: 2), count: 3)

    private var pickCount: Int { photoPicks.count + samplePicks.count }

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()

            VStack(spacing: 14) {
                topBar
                segmentedPill
                switch tab {
                case .photos:
                    photoLibrary
                case .samples:
                    samplesGrid
                }
            }
        }
        .overlay(alignment: .bottom) {
            bottomBar
        }
        .disabled(isStaging)
        .overlay {
            if isStaging {
                stagingHUD
            }
        }
    }

    private var topBar: some View {
        HStack {
            Button("Cancel", action: onCancel)
                .font(.subheadline)
                .foregroundStyle(.white)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(Theme.surfaceElevated, in: Capsule())
                .buttonStyle(.plain)

            Spacer()

            // Skip opens the editor with an empty timeline.
            Button("Skip") { onDone([]) }
                .font(.subheadline)
                .foregroundStyle(.white)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(Theme.surfaceElevated, in: Capsule())
                .buttonStyle(.plain)
        }
        .padding(.horizontal, 16)
        .padding(.top, 8)
    }

    private var segmentedPill: some View {
        HStack(spacing: 4) {
            ForEach(Tab.allCases, id: \.self) { candidate in
                Button {
                    tab = candidate
                } label: {
                    Text(candidate.rawValue)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(tab == candidate ? .white : Theme.textTertiary)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 7)
                        .background {
                            if tab == candidate {
                                Capsule().fill(Theme.surfaceElevated)
                            }
                        }
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .background(Theme.surface, in: Capsule())
    }

    /// The system photo grid, embedded inline. Its own selection chrome is
    /// disabled so the floating "Add N" bar is the single commit point.
    private var photoLibrary: some View {
        PhotosPicker(
            selection: $photoPicks,
            maxSelectionCount: 20,
            selectionBehavior: .continuousAndOrdered,
            matching: .any(of: [.images, .videos]),
            preferredItemEncoding: .current,
            photoLibrary: .shared()
        ) {
            EmptyView()
        }
        .photosPickerStyle(.inline)
        .photosPickerDisabledCapabilities(.selectionActions)
        .photosPickerAccessoryVisibility(.hidden, edges: .bottom)
        .ignoresSafeArea(edges: .bottom)
    }

    private var samplesGrid: some View {
        ScrollView(showsIndicators: false) {
            LazyVGrid(columns: columns, spacing: 2) {
                ForEach(FixtureLibrary.samples, id: \.self) { url in
                    MediaCell(url: url, selectionIndex: sampleIndex(of: url))
                        .onTapGesture { toggleSample(url) }
                }
            }
            .padding(.bottom, 110)
        }
    }

    private var bottomBar: some View {
        HStack {
            Image(systemName: "line.3.horizontal.decrease")
                .font(.system(size: 17, weight: .medium))
                .foregroundStyle(.white)
                .frame(width: 46, height: 46)
                .background(Theme.surfaceElevated, in: Circle())

            Spacer()

            if pickCount == 0 {
                Text("Select Items")
                    .font(.headline)
                    .foregroundStyle(.white)
            } else {
                Button(action: commit) {
                    Text("Add \(pickCount)")
                        .font(.headline)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 28)
                        .padding(.vertical, 12)
                        .background(Theme.accent, in: Capsule())
                }
                .buttonStyle(.plain)
            }

            Spacer()

            Image(systemName: "magnifyingglass")
                .font(.system(size: 17, weight: .medium))
                .foregroundStyle(.white)
                .frame(width: 46, height: 46)
                .background(Theme.surfaceElevated, in: Circle())
        }
        .padding(.horizontal, 18)
        .padding(.bottom, 8)
    }

    private var stagingHUD: some View {
        VStack(spacing: 12) {
            ProgressView()
                .controlSize(.large)
            Text("Preparing media…")
                .font(.subheadline)
                .foregroundStyle(Theme.textSecondary)
        }
        .padding(28)
        .background(Theme.surfaceElevated, in: RoundedRectangle(cornerRadius: 16))
    }

    /// Copy library picks to disk, then hand every URL (system picks in
    /// selection order, then samples) to the caller.
    private func commit() {
        isStaging = true
        let picks = photoPicks
        let samples = samplePicks
        Task { @MainActor in
            var urls: [URL] = []
            for pick in picks {
                if let url = await MediaImporter.stage(pick) {
                    urls.append(url)
                } else {
                    print("cutlass: skipped an unloadable library pick")
                }
            }
            urls.append(contentsOf: samples)
            isStaging = false
            onDone(urls)
        }
    }

    private func sampleIndex(of url: URL) -> Int? {
        samplePicks.firstIndex(of: url).map { $0 + photoPicks.count + 1 }
    }

    private func toggleSample(_ url: URL) {
        if let index = samplePicks.firstIndex(of: url) {
            samplePicks.remove(at: index)
        } else {
            samplePicks.append(url)
        }
    }
}

#Preview {
    MediaPickerView(onCancel: {}, onDone: { _ in })
}
