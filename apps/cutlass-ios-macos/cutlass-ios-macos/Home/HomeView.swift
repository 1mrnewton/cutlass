import SwiftUI

/// Home screen: purple header with quick actions, recent projects row,
/// template carousels, and a floating new-project button.
struct HomeView: View {
    var onNewProject: () -> Void
    var onBlankProject: () -> Void

    var body: some View {
        ZStack(alignment: .top) {
            // Ambient header wash; fades into the background before the fold.
            VStack(spacing: 0) {
                Theme.homeHeader
                    .frame(height: 430)
                Theme.background
            }
            .ignoresSafeArea()

            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: 28) {
                    topBar
                    QuickActionsGrid(
                        onNewProject: onNewProject,
                        onBlankProject: onBlankProject
                    )
                    projectsSection
                    ForEach(MockData.templateSections) { section in
                        TemplateSection(section: section)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 96)
            }
        }
        .overlay(alignment: .bottomTrailing) {
            fab
        }
    }

    private var topBar: some View {
        HStack(spacing: 18) {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Theme.accent)
                .frame(width: 34, height: 34)
                .overlay {
                    Text("Cu")
                        .font(.subheadline.bold())
                        .foregroundStyle(.white)
                }

            Spacer()

            Circle()
                .fill(Theme.premiumBadge)
                .frame(width: 26, height: 26)
                .overlay {
                    Image(systemName: "crown.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(.white)
                }

            Image(systemName: "lightbulb")
                .font(.system(size: 17))
                .foregroundStyle(.white)

            Image(systemName: "ellipsis")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(.white)
        }
        .padding(.top, 6)
    }

    private var projectsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Projects")
                .font(.title3.bold())
                .foregroundStyle(.white)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(alignment: .top, spacing: 12) {
                    ForEach(MockData.projects) { project in
                        Button(action: onNewProject) {
                            ProjectCard(project: project)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    private var fab: some View {
        Button(action: onNewProject) {
            Image(systemName: "plus")
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 56, height: 56)
                .background(Theme.accent, in: Circle())
                .shadow(color: .black.opacity(0.45), radius: 10, y: 4)
        }
        .buttonStyle(.plain)
        .padding(.trailing, 20)
        .padding(.bottom, 12)
    }
}

#Preview {
    HomeView(onNewProject: {}, onBlankProject: {})
}
