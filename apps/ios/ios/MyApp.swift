import OSLog
import SwiftUI

@main struct MyApp: App {
    init() {
        do {
            try start(root: Places.root)
        } catch {
            Logger(subsystem: "cutlass", category: "app").error("\(error.localizedDescription)")
        }
    }

    var body: some Scene {
        WindowGroup {
            AppScreen()
                .preferredColorScheme(.dark)
        }
    }
}
