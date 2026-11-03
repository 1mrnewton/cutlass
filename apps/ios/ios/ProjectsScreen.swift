import SwiftUI

/// Where the app keeps its projects and the files they share.
enum Folders {
    static let projects = URL.documentsDirectory.appending(path: "Projects")
    static let bundle = URL.documentsDirectory.appending(path: "Bundle")
    static let cache = URL.documentsDirectory.appending(path: "Cache")
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

    private let columns = [GridItem(.adaptive(minimum: 160), spacing: 12)]

    var body: some View {
        VStack(spacing: 0) {
            TitleBar(title: "Projects")
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
                            Button { onOpen(project) } label: { ProjectCard(project: project) }
                                .buttonStyle(.plain)
                        }
                    }
                }
                .padding(16)
            }
        }
        .background(Palette.panel)
        .onAppear(perform: reload)
    }

    private func reload() {
        projects = listProjects(root: Folders.projects.path(percentEncoded: false))
    }

    private func newProject() {
        do {
            let path = try createProject(root: Folders.projects.path(percentEncoded: false))
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
}
