//! A clip's keyframes and the Graphs panel's curves, shared with the
//! desktop in the editor crate.

/// A clip's keyframes, as timeline times.
pub use cutlass_editor::ClipKeys as PhoneKeys;
/// A Graphs panel curve to show.
pub use cutlass_editor::KeyGraph as PhoneGraph;
/// The Graphs panel curves, in the panel's order after None.
pub use cutlass_editor::key_graphs as graphs;
