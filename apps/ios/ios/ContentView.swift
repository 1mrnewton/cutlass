import SwiftUI

struct ContentView: View {
    @State private var status = "Opening…"

    var body: some View {
        Text(status)
            .padding()
            .task {
                let documents = URL.documentsDirectory
                do {
                    try FileManager.default.createDirectory(at: documents.appending(path: "Bundle"), withIntermediateDirectories: true)
                    let editor = try Editor.open(
                        project: documents.appending(path: "Project").path(),
                        bundle: documents.appending(path: "Bundle").path(),
                        cache: documents.appending(path: "Cache").path()
                    )
                    status = "Lanes: \(try editor.lanes().count)"
                } catch {
                    status = "\(error)"
                }
            }
    }
}
