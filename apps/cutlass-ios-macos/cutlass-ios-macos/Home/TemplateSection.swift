import SwiftUI

/// A titled carousel of portrait template cards ("Shorts templates",
/// "Lifestyle", ...).
struct TemplateSection: View {
    var section: MockTemplateSection

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(section.title)
                .font(.title3.bold())
                .foregroundStyle(.white)
            Text(section.subtitle)
                .font(.footnote)
                .foregroundStyle(Theme.textSecondary)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(section.templates) { template in
                        TemplateCard(template: template)
                    }
                }
            }
            .padding(.top, 12)
        }
    }
}

private struct TemplateCard: View {
    var template: MockTemplate

    var body: some View {
        MockArtView(art: template.art, symbolSize: 42)
            .frame(width: 150, height: 250)
            .overlay {
                if let caption = template.caption {
                    Text(caption)
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.center)
                        .shadow(color: .black.opacity(0.6), radius: 3)
                        .padding(.horizontal, 14)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

#Preview {
    TemplateSection(section: MockData.templateSections[0])
        .padding()
        .background(Theme.background)
}
