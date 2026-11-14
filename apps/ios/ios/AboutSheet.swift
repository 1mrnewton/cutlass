import SwiftUI

/// About Cutlass: the app's version and license, then every notice Rust
/// lists, the sticker sets first. A notice opens to its full text.
struct AboutSheet: View {
    @Environment(\.dismiss) private var dismiss
    private let list = notices()

    var body: some View {
        NavigationStack {
            List {
                Section {
                    row("Version", Self.version)
                    row("License", cutlassLicense())
                }
                Section("Notices") {
                    ForEach(Array(list.enumerated()), id: \.offset) { index, notice in
                        NavigationLink {
                            NoticePage(notice: notice, index: UInt32(index))
                        } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(notice.title).foregroundStyle(Palette.text)
                                Text(notice.license)
                                    .font(.caption)
                                    .foregroundStyle(Palette.muted)
                                    .lineLimit(2)
                            }
                        }
                    }
                }
                .listRowBackground(Palette.bar)
            }
            .listRowBackground(Palette.bar)
            .scrollContentBackground(.hidden)
            .background(Palette.panel)
            .navigationTitle("About")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .preferredColorScheme(.dark)
        .tint(Palette.accent)
    }

    private func row(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title).foregroundStyle(Palette.text)
            Spacer()
            Text(value).foregroundStyle(Palette.muted)
        }
        .listRowBackground(Palette.bar)
    }

    private static var version: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        return "\(short) (\(build))"
    }
}

/// One notice's full text, as Rust keeps it.
private struct NoticePage: View {
    let notice: Notice
    let index: UInt32

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                if !notice.link.isEmpty, let url = URL(string: notice.link) {
                    Link(notice.link, destination: url).font(.footnote)
                }
                Text(noticeText(index: index))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(Palette.text)
                    .textSelection(.enabled)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
        }
        .background(Palette.panel)
        .navigationTitle(notice.title)
        .navigationBarTitleDisplayMode(.inline)
    }
}
