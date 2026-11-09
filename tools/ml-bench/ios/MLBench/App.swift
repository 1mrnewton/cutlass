import SwiftUI

/// Runs ml-bench once on launch. The reports go to the console and to the
/// app's Documents folder.
@main
struct MLBenchApp: App {
    @State private var status = "Running ml-bench. Keep the app open."

    var body: some Scene {
        WindowGroup {
            Text(status)
                .padding()
                .task {
                    UIApplication.shared.isIdleTimerDisabled = true
                    let code = await Task.detached(priority: .userInitiated) {
                        ml_bench_run_ios()
                    }.value
                    status = code == 0 ? "Done." : "Failed (\(code)). See the console."
                    UIApplication.shared.isIdleTimerDisabled = false
                }
        }
    }
}
