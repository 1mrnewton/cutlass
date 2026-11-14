plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.scytheralpha.cutlass_android"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.scytheralpha.cutlass_android"
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
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
