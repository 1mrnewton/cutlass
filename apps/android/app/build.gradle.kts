plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

// The app carries the workspace version, so the APK of a release says the
// same as its tag, a pre-release such as 0.8.0-rc.1 included. The version
// code grows with major.minor.patch: 0.8.0 is 8000, 1.2.3 is 1002003, so
// each release is above the last in the store's order. A pre-release has
// its release's code; it is a trial of the pipeline and never goes to Play.
val workspaceVersionParts: MatchResult = run {
    val cargo = rootProject.file("../../Cargo.toml").readText()
    Regex("""(?m)^version = "([0-9]+)\.([0-9]+)\.([0-9]+)(-[0-9A-Za-z.-]+)?"""").find(cargo)
        ?: error("the workspace Cargo.toml names no version")
}
val workspaceVersion: String = workspaceVersionParts.value.substringAfter('"').removeSuffix("\"")
val workspaceVersionCode: Int = workspaceVersionParts.groupValues.subList(1, 4).map(String::toInt)
    .let { (major, minor, patch) -> major * 1_000_000 + minor * 1_000 + patch }

android {
    namespace = "com.scytheralpha.cutlass_android"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.scytheralpha.cutlassapp"
        minSdk = 28
        targetSdk = 36
        versionCode = workspaceVersionCode
        versionName = workspaceVersion

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    // The upload key, from the variables the release workflow sets from the
    // repository's secrets. Without them the release build is unsigned and
    // cannot be installed; the debug build keeps its own key either way.
    signingConfigs {
        val keystore = System.getenv("CUTLASS_ANDROID_KEYSTORE")
        if (!keystore.isNullOrEmpty()) {
            create("release") {
                storeFile = file(keystore)
                storePassword = System.getenv("CUTLASS_ANDROID_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("CUTLASS_ANDROID_KEY_ALIAS")
                keyPassword = System.getenv("CUTLASS_ANDROID_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.findByName("release")
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
    // The device tests read the engine's small test pictures.
    sourceSets.getByName("androidTest").assets.srcDir("../../../crates/engine/fixtures")
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    // The uniffi bindings call the Rust library through JNA.
    implementation(libs.jna) { artifact { type = "aar" } }
    implementation(libs.kotlinx.coroutines.android)
    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}
// The Rust library is built by apps/android/build-rust.sh, as the iOS app's
// is by apps/ios/build-rust.sh. Without it the app would crash at launch.
val rustLibrary = layout.projectDirectory.file("src/main/jniLibs/arm64-v8a/libcutlass_ffi_bindings.so")
val checkRustLibrary by tasks.registering {
    val library = rustLibrary.asFile
    doLast {
        check(library.isFile) { "Run apps/android/build-rust.sh first: ${library.path} is missing" }
    }
}
tasks.named("preBuild") { dependsOn(checkRustLibrary) }
