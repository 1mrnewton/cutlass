//
//  ContentView.swift
//  cutlass-ios-macos
//
//  Home screen: a gallery of `.cutlass` projects plus a New Project flow. Tapping
//  a project (or creating one) pushes the editor.
//

import SwiftUI
import ImageIO

/// Where the editor should be pushed to: an existing file (`create == false`)
/// or a fresh project to be created (`create == true`).
struct EditorRoute: Hashable {
    let url: URL
    let create: Bool
    var fpsNum: Int32 = 30
    var fpsDen: Int32 = 1
}

struct ContentView: View {
    @StateObject private var store = ProjectStore()
    @State private var path: [EditorRoute] = []
    @State private var showNewProject = false

    private let columns = [GridItem(.adaptive(minimum: 150), spacing: 16)]

    var body: some View {
        NavigationStack(path: $path) {
            Group {
                if store.projects.isEmpty {
                    emptyState
                } else {
                    gallery
                }
            }
            .navigationTitle("Cutlass")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { showNewProject = true } label: {
                        Label("New Project", systemImage: "plus")
                    }
                }
            }
            .navigationDestination(for: EditorRoute.self) { route in
                if let editor = EditorView(
                    projectURL: route.url, cacheDir: store.cacheDir, create: route.create,
                    fpsNum: route.fpsNum, fpsDen: route.fpsDen
                ) {
                    editor
                } else {
                    ContentUnavailableView(
                        "Couldn't open project",
                        systemImage: "exclamationmark.triangle",
                        description: Text("The editor engine failed to start.")
                    )
                }
            }
            .sheet(isPresented: $showNewProject) {
                NewProjectSheet { fpsNum, fpsDen in/
                    showNewProject = false
                    let url = store.uniqueProjectURL()
                    path.append(EditorRoute(url: url, create: true, fpsNum: fpsNum, fpsDen: fpsDen))
                }
            }
        }
        .onAppear { store.refresh() }
        // Refresh when returning to the root (a project may have been created/saved).
        .onChange(of: path) { _, newPath in
            if newPath.isEmpty { store.refresh() }
        }
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("No Projects", systemImage: "film.stack")
        } description: {
            Text("Create a project to start editing.")
        } actions: {
            Button("New Project") { showNewProject = true }
                .buttonStyle(.borderedProminent)
        }
    }

    private var gallery: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 16) {
                ForEach(store.projects) { entry in
                    Button {
                        path.append(EditorRoute(url: entry.url, create: false))
                    } label: {
                        ProjectCard(entry: entry)
                    }
                    .buttonStyle(.plain)
                    .contextMenu {
                        Button(role: .destructive) { store.delete(entry) } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                }
            }
            .padding()
        }
    }
}

private struct ProjectCard: View {
    let entry: ProjectEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ZStack {
                RoundedRectangle(cornerRadius: 10).fill(Color(white: 0.15))
                if let thumb = entry.thumbnailURL, let image = loadThumbnail(at: thumb) {
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                } else {
                    Image(systemName: "film")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(height: 100)
            .clipShape(RoundedRectangle(cornerRadius: 10))

            Text(entry.name).font(.subheadline).lineLimit(1)
            Text(entry.modified, style: .date)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}

/// New-project options. v1 keeps it to frame rate (aspect stays Auto/follow).
private struct NewProjectSheet: View {
    let onCreate: (Int32, Int32) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var fps = 30

    private let fpsChoices = [24, 30, 60]

    var body: some View {
        NavigationStack {
            Form {
                Picker("Frame rate", selection: $fps) {
                    ForEach(fpsChoices, id: \.self) { Text("\($0) fps").tag($0) }
                }
            }
            .navigationTitle("New Project")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Create") { onCreate(Int32(fps), 1) }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 320, minHeight: 160)
        #endif
    }
}

/// Load a PNG thumbnail file into a SwiftUI `Image` (cross-platform, no UIKit).
func loadThumbnail(at url: URL) -> Image? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else { return nil }
    return Image(cg, scale: 1, label: Text("Thumbnail"))
}

#Preview {
    ContentView()
}
