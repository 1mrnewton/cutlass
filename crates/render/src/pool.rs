//! GPU resources kept from one frame to the next.
//!
//! A frame asks for its textures, uniform buffers and bind groups here
//! instead of creating them. What one frame used is handed out again to the
//! next frame that asks for the same kind, so playing the same layers
//! creates nothing after the first frame. Anything the last frame used that
//! this frame did not ask for is dropped when this frame ends, so the pool
//! never holds more than one frame's worth.
//!
//! Nothing taken during a frame is handed out again in that frame: two
//! layers never share a texture or a buffer before the commands that use
//! them are submitted. After the submit, a `write_texture` or
//! `write_buffer` for the next frame is ordered after the commands already
//! submitted, so reusing them does not change a frame in flight.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

/// One texture's shape and use. Equal keys can share a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TextureKey {
    pub(crate) size: (u32, u32),
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) usage: wgpu::TextureUsages,
}

/// A resource a bind group points at.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Bind<'a> {
    Buffer(&'a wgpu::Buffer),
    View(&'a wgpu::TextureView),
    Sampler(&'a wgpu::Sampler),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Held {
    Buffer(wgpu::Buffer),
    View(wgpu::TextureView),
    Sampler(wgpu::Sampler),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GroupKey {
    layout: wgpu::BindGroupLayout,
    entries: Vec<(u32, Held)>,
}

/// Handles of one kind: spare ones from the last frame, and the ones this
/// frame took.
struct Recycler<K, V> {
    spare: HashMap<K, Vec<V>>,
    taken: Vec<(K, V)>,
}

impl<K: std::hash::Hash + Eq + Clone, V: Clone> Recycler<K, V> {
    fn new() -> Self {
        Self {
            spare: HashMap::new(),
            taken: Vec::new(),
        }
    }

    fn take(&mut self, key: &K, make: impl FnOnce() -> V) -> (V, bool) {
        let (value, made) = match self.spare.get_mut(key).and_then(Vec::pop) {
            Some(value) => (value, false),
            None => (make(), true),
        };
        self.taken.push((key.clone(), value.clone()));
        (value, made)
    }

    /// This frame's handles become the next frame's spares. The spares it
    /// did not use are dropped. Each list is reversed so `pop` hands them
    /// out in the order this frame took them: the same layer gets the same
    /// texture and buffer, and so the same bind group, every frame.
    fn recycle(&mut self) {
        self.spare.clear();
        for (key, value) in self.taken.drain(..) {
            self.spare.entry(key).or_default().push(value);
        }
        for list in self.spare.values_mut() {
            list.reverse();
        }
    }
}

struct State {
    textures: Recycler<TextureKey, (wgpu::Texture, wgpu::TextureView)>,
    buffers: Recycler<u64, wgpu::Buffer>,
    /// Bind groups this frame used, and the last frame's.
    groups: HashMap<GroupKey, wgpu::BindGroup>,
    last_groups: HashMap<GroupKey, wgpu::BindGroup>,
    created: u64,
}

/// The resources [`crate::Renderer`] reuses between frames.
pub(crate) struct Pool {
    state: Mutex<State>,
}

impl Pool {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(State {
                textures: Recycler::new(),
                buffers: Recycler::new(),
                groups: HashMap::new(),
                last_groups: HashMap::new(),
                created: 0,
            }),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        // A panic while the lock was held leaves handles that are still
        // valid; the worst case is one resource created again.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A texture of `key`'s shape and use, with its whole view. Its
    /// contents are whatever the last frame left.
    pub(crate) fn texture(
        &self,
        device: &wgpu::Device,
        label: &'static str,
        key: TextureKey,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let mut state = self.state();
        let (pair, made) = state.textures.take(&key, || {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: key.size.0,
                    height: key.size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: key.format,
                usage: key.usage,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        });
        state.created += u64::from(made);
        pair
    }

    /// A uniform buffer of `size` bytes that starts with `bytes`.
    pub(crate) fn uniform(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &'static str,
        size: u64,
        bytes: &[u8],
    ) -> wgpu::Buffer {
        let mut state = self.state();
        let (buffer, made) = state.buffers.take(&size, || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        state.created += u64::from(made);
        drop(state);
        queue.write_buffer(&buffer, 0, bytes);
        buffer
    }

    /// A bind group of `layout` over `entries`, each a binding number and
    /// what it points at. The same layout and resources give the same group.
    pub(crate) fn group(
        &self,
        device: &wgpu::Device,
        label: &'static str,
        layout: &wgpu::BindGroupLayout,
        entries: &[(u32, Bind<'_>)],
    ) -> wgpu::BindGroup {
        let key = GroupKey {
            layout: layout.clone(),
            entries: entries
                .iter()
                .map(|(binding, bind)| {
                    let held = match bind {
                        Bind::Buffer(buffer) => Held::Buffer((*buffer).clone()),
                        Bind::View(view) => Held::View((*view).clone()),
                        Bind::Sampler(sampler) => Held::Sampler((*sampler).clone()),
                    };
                    (*binding, held)
                })
                .collect(),
        };
        let mut state = self.state();
        if let Some(group) = state.groups.get(&key) {
            return group.clone();
        }
        let group = match state.last_groups.remove(&key) {
            Some(group) => group,
            None => {
                state.created += 1;
                let resources: Vec<wgpu::BindGroupEntry<'_>> = entries
                    .iter()
                    .map(|(binding, bind)| wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: match bind {
                            Bind::Buffer(buffer) => buffer.as_entire_binding(),
                            Bind::View(view) => wgpu::BindingResource::TextureView(view),
                            Bind::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
                        },
                    })
                    .collect();
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout,
                    entries: &resources,
                })
            }
        };
        state.groups.insert(key, group.clone());
        group
    }

    /// Ends a frame whose commands are submitted.
    pub(crate) fn recycle(&self) {
        let mut state = self.state();
        state.textures.recycle();
        state.buffers.recycle();
        state.last_groups = std::mem::take(&mut state.groups);
    }

    /// Textures, buffers and bind groups created so far.
    #[cfg(test)]
    pub(crate) fn created(&self) -> u64 {
        self.state().created
    }
}
