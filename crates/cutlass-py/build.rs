fn main() {
    // On macOS the extension module must leave CPython symbols undefined for
    // the host interpreter to resolve at import time. Emitting the link args
    // from our own build script covers every build path — including sdist
    // builds, where cargo runs from the sdist root and never discovers this
    // crate's `.cargo/config.toml`.
    pyo3_build_config::add_extension_module_link_args();
}
