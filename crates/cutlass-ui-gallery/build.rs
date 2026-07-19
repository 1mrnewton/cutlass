fn main() {
    // FlexboxLayout is experimental in Slint 1.16 (same as cutlass-desktop).
    unsafe {
        std::env::set_var("SLINT_ENABLE_EXPERIMENTAL_FEATURES", "1");
    }
    slint_build::compile("ui/app.slint").unwrap();
}
