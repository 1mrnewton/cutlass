// swift-tools-version: 5.9
import PackageDescription

// Local package wrapping the Rust `cutlass-mobile` engine for the iOS/macOS app.
//
// Add it to the app with: File > Add Package Dependencies… > Add Local… and
// pick this `CutlassMobile` folder, then add the `CutlassMobile` library to the
// app target. The binary target carries the prebuilt static lib; the Swift
// target links the system frameworks wgpu/Metal interop needs.
let package = Package(
    name: "CutlassMobile",
    platforms: [
        .iOS(.v15),
        .macOS(.v13),
    ],
    products: [
        .library(name: "CutlassMobile", targets: ["CutlassMobile"]),
    ],
    targets: [
        .binaryTarget(
            name: "CutlassMobileFFI",
            path: "CutlassMobileFFI.xcframework"
        ),
        .target(
            name: "CutlassMobile",
            dependencies: ["CutlassMobileFFI"],
            resources: [
                .copy("Resources/sample.mp4"),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("CoreFoundation"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("QuartzCore"),
                .linkedFramework("IOSurface"),
                .linkedFramework("Foundation"),
                // Decoder (AVAssetReader + VideoToolbox).
                .linkedFramework("AVFoundation"),
                .linkedFramework("VideoToolbox"),
                .linkedFramework("AudioToolbox"),
                .linkedLibrary("c++"),
            ]
        ),
        .testTarget(
            name: "CutlassMobileTests",
            dependencies: ["CutlassMobile"]
        ),
    ]
)
