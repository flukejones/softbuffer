//! Interface implemented by backends

use crate::{AlphaMode, InitError, Pixel, PresentMode, Rect, SoftBufferError};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::num::NonZeroU32;

pub(crate) trait ContextInterface<D: HasDisplayHandle + ?Sized> {
    fn new(display: D) -> Result<Self, InitError<D>>
    where
        D: Sized,
        Self: Sized;
}

pub(crate) trait SurfaceInterface<D: HasDisplayHandle + ?Sized, W: HasWindowHandle + ?Sized> {
    type Context: ContextInterface<D>;
    type Buffer<'surface>: BufferInterface
    where
        Self: 'surface;

    fn new(window: W, context: &Self::Context) -> Result<Self, InitError<W>>
    where
        W: Sized,
        Self: Sized;

    /// Get the inner window handle.
    fn window(&self) -> &W;

    fn supports_alpha_mode(&self, alpha_mode: AlphaMode) -> bool;

    /// Reconfigure the internal buffer(s).
    fn configure(
        &mut self,
        width: NonZeroU32,
        height: NonZeroU32,
        alpha_mode: AlphaMode,
    ) -> Result<(), SoftBufferError>;

    /// Get the next buffer to render into.
    fn next_buffer(&mut self, alpha_mode: AlphaMode) -> Result<Self::Buffer<'_>, SoftBufferError>;

    /// Set how presenting is synchronized with the display.
    ///
    /// Backends without vsync control ignore this (effectively always `Immediate`).
    fn set_present_mode(&mut self, _present_mode: PresentMode) {}

    /// Fetch the buffer from the window.
    fn fetch(&mut self) -> Result<Vec<Pixel>, SoftBufferError> {
        Err(SoftBufferError::Unimplemented)
    }
}

pub(crate) trait BufferInterface {
    fn byte_stride(&self) -> NonZeroU32;
    fn width(&self) -> NonZeroU32;
    fn height(&self) -> NonZeroU32;
    fn pixels_mut(&mut self) -> &mut [Pixel];
    fn age(&self) -> u8;
    fn present_with_damage(self, damage: &[Rect]) -> Result<(), SoftBufferError>;
}
