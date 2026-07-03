import SwiftUI

/// A recent-project card in the horizontally scrolling Projects row.
struct ProjectCard: View {
    var project: MockProject

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            MockArtView(art: project.art, symbolSize: 26)
                .frame(width: 138, height: 92)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(alignment: .topLeading) {
                    DurationBadge(duration: project.duration)
                        .padding(6)
                }
                .overlay(alignment: .bottomTrailing) {
                    Image(systemName: "ellipsis")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(width: 22, height: 22)
                        .background(.black.opacity(0.45), in: Circle())
                        .padding(6)
                }

            Text(project.dateLabel)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
        }
    }
}

#Preview {
    ProjectCard(project: MockData.projects[0])
        .padding()
        .background(Theme.background)
}
