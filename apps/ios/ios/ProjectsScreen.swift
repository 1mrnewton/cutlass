import SwiftUI

/// The two paths the app hands to Rust. Rust lays out everything under the
/// root: the projects, the cache, the downloads, the proxies, the logs.
enum Places {
    /// The one Cutlass folder, private to the app.
    static let root = URL.applicationSupportDirectory
        .appending(path: "Cutlass").path(percentEncoded: false)
    /// The app's read-only resources.
    static let bundle = Bundle.main.resourceURL?.path(percentEncoded: false) ?? root
}

/// The whole app: the project list, or one open project in the editor.
/// There is no stack to go back through; the editor's home button returns
/// here, and the list is read again.
struct AppScreen: View {
    @State private var open: ProjectSummary?

    var body: some View {
        Group {
            if let project = open {
                EditorScreen(project: project) { open = nil }
                    .id(project.path)
                    .transition(.opacity)
            } else {
                ProjectsScreen { open = $0 }
                    .transition(.opacity)
            }
        }
        .animation(.easeOut(duration: 0.2), value: open?.path)
    }
}

/// The launch screen: a new-project button and the saved projects, newest
/// first, as on the desktop. A tap opens the project.
struct ProjectsScreen: View {
    let onOpen: (ProjectSummary) -> Void
    @State private var projects: [ProjectSummary] = []
    @State private var failure: String?
    /// The project being renamed, and the name typed so far.
    @State private var renaming: ProjectSummary?
    @State private var newName = ""
    /// The project waiting for its delete to be confirmed.
    @State private var deleting: ProjectSummary?
    @State private var showsAbout = false

    private let columns = [GridItem(.adaptive(minimum: 160), spacing: 12)]

    var body: some View {
        VStack(spacing: 0) {
            TitleBar(title: "Projects")
                .overlay(alignment: .trailing) {
                    Button { showsAbout = true } label: {
                        HugeIcon("settings", size: 20)
                            .foregroundStyle(Palette.text)
                            .frame(width: 44, height: 44)
                            .contentShape(Rectangle())
                    }
                    .padding(.trailing, 6)
                    .accessibilityLabel("About")
                }
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Button(action: newProject) {
                        Text("New project")
                            .font(.body.weight(.semibold))
                            .foregroundStyle(Palette.exportInk)
                            .frame(maxWidth: .infinity)
                            .frame(height: 48)
                            .background(Palette.export, in: RoundedRectangle(cornerRadius: 8))
                    }
                    if let failure {
                        Text(failure)
                            .font(.footnote)
                            .foregroundStyle(.red)
                    }
                    if projects.isEmpty {
                        Text("No projects yet.")
                            .font(.subheadline)
                            .foregroundStyle(Palette.muted)
                    }
                    LazyVGrid(columns: columns, spacing: 12) {
                        ForEach(projects, id: \.path) { project in
                            ZStack(alignment: .topTrailing) {
                                Button { onOpen(project) } label: { ProjectCard(project: project) }
                                    .buttonStyle(.plain)
                                    .contextMenu { actions(for: project) }
                                Menu { actions(for: project) } label: {
                                    HugeIcon("more", size: 18)
                                        .foregroundStyle(Palette.muted)
                                        .frame(width: 40, height: 36)
                                        .contentShape(Rectangle())
                                }
                                .accessibilityLabel("More for \(project.name)")
                            }
                        }
                    }
                }
                .padding(16)
            }
        }
        .background(Palette.panel)
        .onAppear(perform: reload)
        // The dialogs hand their project to the buttons: the binding is
        // cleared before a button's action runs.
        .alert(
            "Rename project",
            isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } }),
            presenting: renaming
        ) { project in
            TextField("Name", text: $newName)
            Button("Cancel", role: .cancel) {}
            Button("Rename") { rename(project) }
        }
        .confirmationDialog(
            "Delete \(deleting?.name ?? "")?",
            isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } }),
            titleVisibility: .visible,
            presenting: deleting
        ) { project in
            Button("Delete", role: .destructive) { delete(project) }
            Button("Cancel", role: .cancel) {}
        } message: { _ in
            Text("The project and its imported media are removed from this phone.")
        }
        .sheet(isPresented: $showsAbout) { AboutSheet() }
    }

    @ViewBuilder private func actions(for project: ProjectSummary) -> some View {
        Button {
            newName = project.name
            renaming = project
        } label: { Label("Rename", image: HugeIcon.asset("rename")) }
        Button { duplicate(project) } label: {
            Label("Duplicate", image: HugeIcon.asset("duplicate"))
        }
        Button(role: .destructive) { deleting = project } label: {
            Label("Delete", image: HugeIcon.asset("delete"))
        }
    }

    /// Run a project change in Rust, showing its refusal, then read the list again.
    private func change(_ body: () throws -> Void) {
        do {
            try body()
            failure = nil
        } catch let EditorError.Failed(message) {
            failure = message
        } catch {
            failure = error.localizedDescription
        }
        reload()
    }

    private func rename(_ project: ProjectSummary) {
        let name = newName
        change { try renameProject(root: Places.root, project: project.path, name: name) }
    }

    private func duplicate(_ project: ProjectSummary) {
        change { _ = try duplicateProject(root: Places.root, project: project.path) }
    }

    private func delete(_ project: ProjectSummary) {
        change { try deleteProject(root: Places.root, project: project.path) }
    }

    private func reload() {
        projects = listProjects(root: Places.root)
    }

    private func newProject() {
        do {
            let path = try createProject(root: Places.root)
            failure = nil
            reload()
            if let made = projects.first(where: { $0.path == path }) {
                onOpen(made)
            }
        } catch let EditorError.Failed(message) {
            failure = message
        } catch {
            failure = error.localizedDescription
        }
    }
}

/// A slim bar with a centered title, and an optional button on the left.
struct TitleBar<Leading: View>: View {
    let title: String
    let background: Color
    @ViewBuilder let leading: Leading

    init(title: String, background: Color = Palette.bar, @ViewBuilder leading: () -> Leading) {
        self.title = title
        self.background = background
        self.leading = leading()
    }

    var body: some View {
        ZStack {
            Text(title)
                .font(.headline)
                .foregroundStyle(Palette.text)
                .lineLimit(1)
                .padding(.horizontal, 60)
            HStack {
                leading
                Spacer()
            }
            .padding(.horizontal, 8)
        }
        .frame(height: 48)
        .frame(maxWidth: .infinity)
        .background(background.ignoresSafeArea(edges: .top))
    }
}

extension TitleBar where Leading == EmptyView {
    init(title: String, background: Color = Palette.bar) {
        self.init(title: title, background: background) { EmptyView() }
    }
}

/// One saved project: its name, then its length and the day it changed.
struct ProjectCard: View {
    let project: ProjectSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(project.name)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(Palette.text)
                .lineLimit(1)
                // Room for the card's more button.
                .padding(.trailing, 24)
            Text(meta)
                .font(.caption)
                .foregroundStyle(Palette.muted)
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .padding(14)
        .background(Palette.bar, in: RoundedRectangle(cornerRadius: 12))
        .contentShape(RoundedRectangle(cornerRadius: 12))
    }

    private var meta: String {
        let day = Self.day.string(
            from: Date(timeIntervalSince1970: Double(project.modifiedMs) / 1000))
        return project.durationUs > 0 ? "\(Self.length(project.durationUs))  \(day)" : day
    }

    /// "4.2s" under a minute, "1:05.3" from a minute.
    static func length(_ micros: Int64) -> String {
        let millis = micros / 1_000
        let minutes = millis / 60_000
        let whole = millis % 60_000 / 1_000
        let tenth = millis % 1_000 / 100
        return minutes > 0
            ? String(format: "%lld:%02lld.%lld", minutes, whole, tenth)
            : "\(whole).\(tenth)s"
    }

    private static let day: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter
    }()
}

/// The desktop's colors for the launch screen.
enum Palette {
    static let panel = Color(red: 0x1c / 255, green: 0x1c / 255, blue: 0x1c / 255)
    static let bar = Color(red: 0x26 / 255, green: 0x26 / 255, blue: 0x26 / 255)
    static let export = Color(red: 0x00 / 255, green: 0xc8 / 255, blue: 0xd7 / 255)
    static let exportInk = Color(red: 0x06 / 255, green: 0x22 / 255, blue: 0x26 / 255)
    static let text = Color(red: 0xf3 / 255, green: 0xf3 / 255, blue: 0xf3 / 255)
    static let muted = Color(red: 0xa3 / 255, green: 0xa3 / 255, blue: 0xa3 / 255)
    /// Sliders, the selected control, and other live accents.
    static let accent = export
}
