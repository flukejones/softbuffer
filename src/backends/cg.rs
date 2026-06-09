//! Softbuffer implementation using CoreGraphics.
use crate::error::InitError;
use crate::{backend_interface::*, AlphaMode};
use crate::{Pixel, Rect, SoftBufferError};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool};
use objc2::{define_class, msg_send, AllocAnyThread, DefinedClass, MainThreadMarker, Message};
use objc2_core_foundation::{CFMutableDictionary, CFNumber, CFRetained, CFString, CFType, CGPoint};
use objc2_core_graphics::CGColorSpace;
use objc2_foundation::{
    ns_string, NSDictionary, NSKeyValueChangeKey, NSKeyValueChangeNewKey,
    NSKeyValueObservingOptions, NSNumber, NSObject, NSObjectNSKeyValueObserverRegistration,
    NSString, NSValue,
};
use objc2_io_surface::{
    kIOSurfaceBytesPerElement, kIOSurfaceCacheMode, kIOSurfaceColorSpace, kIOSurfaceHeight,
    kIOSurfaceMapWriteCombineCache, kIOSurfacePixelFormat, kIOSurfaceWidth, IOSurfaceLockOptions,
    IOSurfaceRef,
};
use objc2_quartz_core::{kCAFilterNearest, kCAGravityResize, CALayer, CATransaction};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawWindowHandle};
use tracing::trace;

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::{size_of, ManuallyDrop};
use std::num::NonZeroU32;
use std::ops::Deref;
use std::ptr;
use std::slice;
use std::time::{Duration, Instant};

/// Number of buffers we rotate through (triple-buffering).
///
/// This is what QuartzCore / the compositor seems to require: the front buffer is assigned to
/// `CALayer.contents`, the middle buffer may be what the compositor is currently drawing from
/// (assuming a 1 frame delay), and the back buffer is what we draw into.
const BUFFER_COUNT: usize = 3;

/// How long `next_buffer` waits for the compositor to release the back buffer before giving up and
/// drawing into it anyway.
///
/// This must be bounded: `next_buffer` runs on the main thread, which the compositor also needs to
/// make progress, so waiting indefinitely would deadlock. With triple-buffering the back buffer is
/// almost always already free, so this wait rarely triggers, and proceeding after the timeout only
/// risks tearing in the pathological case where the compositor holds all buffers.
const BACK_BUFFER_WAIT_TIMEOUT: Duration = Duration::from_millis(10);

define_class!(
    #[unsafe(super(NSObject))]
    #[name = "SoftbufferObserver"]
    #[ivars = SendCALayer]
    #[derive(Debug)]
    struct Observer;

    /// NSKeyValueObserving
    impl Observer {
        #[unsafe(method(observeValueForKeyPath:ofObject:change:context:))]
        fn observe_value(
            &self,
            key_path: Option<&NSString>,
            _object: Option<&AnyObject>,
            change: Option<&NSDictionary<NSKeyValueChangeKey, AnyObject>>,
            _context: *mut c_void,
        ) {
            self.update(key_path, change);
        }
    }
);

impl Observer {
    fn new(layer: &CALayer) -> Retained<Self> {
        let this = Self::alloc().set_ivars(SendCALayer(layer.retain()));
        unsafe { msg_send![super(this), init] }
    }

    fn update(
        &self,
        key_path: Option<&NSString>,
        change: Option<&NSDictionary<NSKeyValueChangeKey, AnyObject>>,
    ) {
        let layer = self.ivars();

        let change =
            change.expect("requested a change dictionary in `addObserver`, but none was provided");
        let new = change
            .objectForKey(unsafe { NSKeyValueChangeNewKey })
            .expect("requested change dictionary did not contain `NSKeyValueChangeNewKey`");

        // NOTE: Setting these values usually causes a quarter second animation to occur, which is
        // undesirable.
        //
        // However, since we're setting them inside an observer, there already is a transaction
        // ongoing, and as such we don't need to wrap this in a `CATransaction` ourselves.

        if key_path == Some(ns_string!("contentsScale")) {
            let new = new.downcast::<NSNumber>().unwrap();
            let scale_factor = new.as_cgfloat();

            // Set the scale factor of the layer to match the root layer when it changes (e.g. if
            // moved to a different monitor, or monitor settings changed).
            layer.setContentsScale(scale_factor);
        } else if key_path == Some(ns_string!("bounds")) {
            let new = new.downcast::<NSValue>().unwrap();
            let bounds = new.get_rect().expect("new bounds value was not CGRect");

            // Set `bounds` and `position` so that the new layer is inside the superlayer.
            //
            // This differs from just setting the `bounds`, as it also takes into account any
            // translation that the superlayer may have that we'd want to preserve.
            layer.setFrame(bounds);
        } else {
            panic!("unknown observed keypath {key_path:?}");
        }
    }
}

#[derive(Debug)]
pub struct CGImpl<D, W> {
    /// Our layer.
    layer: SendCALayer,
    /// The layer that our layer was created from.
    ///
    /// Can also be retrieved from `layer.superlayer()`.
    root_layer: SendCALayer,
    observer: Retained<Observer>,
    color_space: CFRetained<CGColorSpace>,
    /// The buffers we render into, rotated on each present.
    ///
    /// The `IOSurface` is shared zero-copy with the compositor (unlike the previous `CGDataProvider`
    /// implementation, where QuartzCore copied internally), so we cannot draw into a surface the
    /// compositor is still reading. We use triple-buffering, which gives the compositor enough
    /// headroom that the buffer we're about to draw into (`buffers.last()`, which was last presented
    /// two frames ago) is almost always free:
    /// - `buffers[0]` and `buffers[1]` may still be referenced by the compositor.
    /// - `buffers[2]` (the back buffer) is what we draw into.
    ///
    /// On present we set `CALayer.contents` to the back buffer and rotate, so it becomes the front.
    ///
    /// NOTE: `next_buffer` only ever waits on `is_in_use` for a bounded time before proceeding
    /// anyway, since it runs on the main thread that the compositor needs to make progress, so
    /// blocking indefinitely would deadlock.
    buffers: Vec<Buffer>,
    /// The width of the buffers.
    width: u32,
    /// The height of the buffers.
    height: u32,
    window_handle: W,
    _display: PhantomData<D>,
}

impl<D, W> Drop for CGImpl<D, W> {
    fn drop(&mut self) {
        // SAFETY: Registered in `new`, must be removed before the observer is deallocated.
        unsafe {
            self.root_layer
                .removeObserver_forKeyPath(&self.observer, ns_string!("contentsScale"));
            self.root_layer
                .removeObserver_forKeyPath(&self.observer, ns_string!("bounds"));
        }

        // Remove the layer we created from the root layer.
        self.layer.removeFromSuperlayer();
    }
}

impl<D: HasDisplayHandle, W: HasWindowHandle> SurfaceInterface<D, W> for CGImpl<D, W> {
    type Context = D;
    type Buffer<'surface>
        = BufferImpl<'surface>
    where
        Self: 'surface;

    fn new(window_src: W, _display: &D) -> Result<Self, InitError<W>> {
        // `NSView`/`UIView` can only be accessed from the main thread.
        let _mtm = MainThreadMarker::new().ok_or(SoftBufferError::PlatformError(
            Some("can only access Core Graphics handles from the main thread".to_string()),
            None,
        ))?;

        let root_layer = match window_src.window_handle()?.as_raw() {
            RawWindowHandle::AppKit(handle) => {
                // SAFETY: The pointer came from `WindowHandle`, which ensures that the
                // `AppKitWindowHandle` contains a valid pointer to an `NSView`.
                //
                // We use `NSObject` here to avoid importing `objc2-app-kit`.
                let view: &NSObject = unsafe { handle.ns_view.cast().as_ref() };

                // Force the view to become layer backed
                let _: () = unsafe { msg_send![view, setWantsLayer: Bool::YES] };

                // SAFETY: `-[NSView layer]` returns an optional `CALayer`
                let layer: Option<Retained<CALayer>> = unsafe { msg_send![view, layer] };
                layer.expect("failed making the view layer-backed")
            }
            RawWindowHandle::UiKit(handle) => {
                // SAFETY: The pointer came from `WindowHandle`, which ensures that the
                // `UiKitWindowHandle` contains a valid pointer to an `UIView`.
                //
                // We use `NSObject` here to avoid importing `objc2-ui-kit`.
                let view: &NSObject = unsafe { handle.ui_view.cast().as_ref() };

                // SAFETY: `-[UIView layer]` returns `CALayer`
                let layer: Retained<CALayer> = unsafe { msg_send![view, layer] };
                layer
            }
            _ => return Err(InitError::Unsupported(window_src)),
        };

        // Add a sublayer, to avoid interfering with the root layer, since setting the contents of
        // e.g. a view-controlled layer is brittle.
        //
        // This layer is removed from the root layer when the surface is `Drop`ped.
        let layer = CALayer::new();
        root_layer.addSublayer(&layer);

        // Set the anchor point and geometry. Softbuffer's uses a coordinate system with the origin
        // in the top-left corner.
        //
        // NOTE: This doesn't really matter unless we start modifying the `position` of our layer
        // ourselves, but it's nice to have in place.
        layer.setAnchorPoint(CGPoint::new(0.0, 0.0));
        layer.setGeometryFlipped(true);

        // Do not use auto-resizing mask.
        //
        // This is done to work around a bug in macOS 14 and above, where views using auto layout
        // may end up setting fractional values as the bounds, and that in turn doesn't propagate
        // properly through the auto-resizing mask and with contents gravity.
        //
        // Instead, we keep the bounds of the layer in sync with the root layer using an observer,
        // see below.
        //
        // layer.setAutoresizingMask(kCALayerHeightSizable | kCALayerWidthSizable);

        let observer = Observer::new(&layer);
        // Observe changes to the root layer's bounds and scale factor, and apply them to our layer.
        //
        // The previous implementation updated the scale factor inside `resize`, but this works
        // poorly with transactions, and is generally inefficient. Instead, we update the scale
        // factor only when needed because the super layer's scale factor changed.
        //
        // Note that inherent in this is an explicit design decision: We control the `bounds` and
        // `contentsScale` of the layer directly, and instead let the `resize` call that the user
        // controls only be the size of the underlying buffer.
        //
        // SAFETY: Observer deregistered in `Drop` before the observer object is deallocated.
        unsafe {
            root_layer.addObserver_forKeyPath_options_context(
                &observer,
                ns_string!("contentsScale"),
                NSKeyValueObservingOptions::New | NSKeyValueObservingOptions::Initial,
                ptr::null_mut(),
            );
            root_layer.addObserver_forKeyPath_options_context(
                &observer,
                ns_string!("bounds"),
                NSKeyValueObservingOptions::New | NSKeyValueObservingOptions::Initial,
                ptr::null_mut(),
            );
        }

        // Stretch the content to fill the surface if it does not have the same size, using
        // nearest-neighbour filtering so scaled buffers stay crisp. See #177.
        layer.setContentsGravity(unsafe { kCAGravityResize });
        layer.setMagnificationFilter(unsafe { kCAFilterNearest });
        layer.setMinificationFilter(unsafe { kCAFilterNearest });

        // Default alpha mode is opaque.
        layer.setOpaque(true);

        // The color space we're using. Initialize it here to reduce work later on.
        // TODO: Allow setting this to something else?
        let color_space = CGColorSpace::new_device_rgb().unwrap();

        // Grab initial width and height from the layer (whose properties have just been initialized
        // by the observer using `NSKeyValueObservingOptionInitial`).
        let size = layer.bounds().size;
        let scale_factor = layer.contentsScale();
        let width = (size.width * scale_factor) as u32;
        let height = (size.height * scale_factor) as u32;

        let buffers = (0..BUFFER_COUNT)
            .map(|_| Buffer::new(width, height, &color_space))
            .collect();

        Ok(Self {
            layer: SendCALayer(layer),
            root_layer: SendCALayer(root_layer),
            observer,
            color_space,
            buffers,
            width,
            height,
            _display: PhantomData,
            window_handle: window_src,
        })
    }

    #[inline]
    fn window(&self) -> &W {
        &self.window_handle
    }

    #[inline]
    fn supports_alpha_mode(&self, alpha_mode: AlphaMode) -> bool {
        // IOSurface doesn't support `Ignored` nor `Postmultiplied`.
        matches!(alpha_mode, AlphaMode::Opaque | AlphaMode::Premultiplied)
    }

    fn configure(
        &mut self,
        width: NonZeroU32,
        height: NonZeroU32,
        alpha_mode: AlphaMode,
    ) -> Result<(), SoftBufferError> {
        let opaque = match alpha_mode {
            AlphaMode::Opaque => true,
            AlphaMode::Premultiplied => false,
            AlphaMode::Ignored | AlphaMode::Postmultiplied => {
                unreachable!("unsupported alpha mode")
            }
        };
        self.layer.setOpaque(opaque);
        // TODO: Set opaque-ness on root layer too? Is that our responsibility, or Winit's?
        // self.root_layer.setOpaque(opaque);

        let width = width.get();
        let height = height.get();

        // TODO: Is this check desirable?
        if self.width == width && self.height == height {
            return Ok(());
        }

        // Recreate buffers. It's fine to release the old ones, `CALayer.contents` and/or the
        // compositor is going to keep a reference to them around as long as they're still in use.
        self.buffers = (0..BUFFER_COUNT)
            .map(|_| Buffer::new(width, height, &self.color_space))
            .collect();
        self.width = width;
        self.height = height;

        Ok(())
    }

    fn next_buffer(&mut self, _alpha_mode: AlphaMode) -> Result<BufferImpl<'_>, SoftBufferError> {
        // We draw into the back buffer (`buffers.last()`) while the compositor reads the others.
        //
        // The back buffer was last presented two frames ago, so with triple-buffering the
        // compositor is almost always done with it. But if the application renders faster than the
        // display refreshes, it might still be in use, and since the `IOSurface` is shared zero-copy
        // with the compositor, writing into it then would risk tearing. So we wait for it to be
        // released first.
        //
        // This wait is bounded: `next_buffer` runs on the main thread that the compositor needs to
        // make progress, so we must not block indefinitely. After the timeout we proceed anyway,
        // accepting a small tearing risk over a deadlock.
        let back = self.buffers.last().unwrap();
        if back.surface.is_in_use() {
            let now = Instant::now();
            while back.surface.is_in_use() {
                if BACK_BUFFER_WAIT_TIMEOUT < now.elapsed() {
                    trace!(
                        "compositor still holding all buffers after {BACK_BUFFER_WAIT_TIMEOUT:?}, \
                         drawing into the back buffer anyway (you might be rendering faster than \
                         the display refreshes)"
                    );
                    break;
                }
                std::thread::yield_now();
            }
        }

        // Lock the back buffer to allow writing to it.
        //
        // Either unlocked in `BufferImpl`s `Drop` or `present_with_damage`.
        self.buffers.last().unwrap().lock();

        Ok(BufferImpl {
            buffers: &mut self.buffers,
            layer: &mut self.layer,
        })
    }
}

/// The implementation used for presenting the back buffer to the surface.
#[derive(Debug)]
pub struct BufferImpl<'surface> {
    buffers: &'surface mut Vec<Buffer>,
    layer: &'surface mut SendCALayer,
}

impl Drop for BufferImpl<'_> {
    fn drop(&mut self) {
        // Unlock the back buffer we locked in `next_buffer`.
        self.buffers.last().unwrap().unlock();
    }
}

impl BufferInterface for BufferImpl<'_> {
    fn byte_stride(&self) -> NonZeroU32 {
        // Use the surface's actual row stride, which may be padded for alignment (a multiple of the
        // cache line size, which is `64` on x86_64 and `128` on Aarch64).
        NonZeroU32::new(self.buffers.last().unwrap().surface.bytes_per_row() as u32).unwrap()
    }

    fn width(&self) -> NonZeroU32 {
        NonZeroU32::new(self.buffers.last().unwrap().surface.width() as u32).unwrap()
    }

    fn height(&self) -> NonZeroU32 {
        NonZeroU32::new(self.buffers.last().unwrap().surface.height() as u32).unwrap()
    }

    fn pixels_mut(&mut self) -> &mut [Pixel] {
        // SAFETY: The back surface is locked in `next_buffer`, so we know it's not being used
        // elsewhere.
        unsafe { self.buffers.last_mut().unwrap().data() }
    }

    fn age(&self) -> u8 {
        self.buffers.last().unwrap().age
    }

    fn present_with_damage(self, _damage: &[Rect]) -> Result<(), SoftBufferError> {
        // Unlock the back buffer now (and not in `Drop`).
        //
        // Note that unlocking effectively flushes the changes, without this, the contents might not
        // be visible to the compositor.
        let this = &mut *ManuallyDrop::new(self);
        let buffers = &mut *this.buffers;
        let layer = &mut *this.layer;
        buffers.last().unwrap().unlock();

        // The CALayer has a default action associated with a change in the layer contents, causing
        // a quarter second fade transition to happen every time a new buffer is applied. This can
        // be avoided by wrapping the operation in a transaction and disabling all actions.
        CATransaction::begin();
        CATransaction::setDisableActions(true);

        // SAFETY: We set `CALayer.contents` to an `IOSurface`, which is an undocumented option, but
        // it's done in browsers and GDK:
        // https://gitlab.gnome.org/GNOME/gtk/-/blob/4266c3c7b15299736df16c9dec57cd8ec7c7ebde/gdk/macos/GdkMacosTile.c#L44
        // And tested to work at least as far back as macOS 10.12.
        unsafe { layer.setContents(Some(buffers.last().unwrap().surface.as_ref())) };

        // Rotate the buffers so the just-presented back buffer becomes the front buffer (which the
        // compositor reads), and the buffer presented two frames ago becomes the new back buffer.
        buffers.rotate_right(1);

        // The new front buffer's contents have just been set by the user.
        let (front, rest) = buffers.split_first_mut().unwrap();
        front.age = 1;
        // Bump the age of the other buffers (older frames).
        for buffer in rest {
            if buffer.age != 0 {
                buffer.age += 1;
            }
        }

        CATransaction::commit();
        Ok(())
    }
}

/// One of the buffers we rotate through.
///
/// Buffers are backed by an `IOSurface`, which is a shared memory buffer that can be passed to the
/// compositor without copying. The best official documentation I've found for how this works is
/// probably this keynote:
/// <https://nonstrict.eu/wwdcindex/wwdc2010/422/>
///
/// The first ~10mins of this keynote is also pretty good, it describes CA and the render server:
/// <https://nonstrict.eu/wwdcindex/wwdc2014/419/>
/// <https://wwdcnotes.com/documentation/wwdcnotes/wwdc14-419-advanced-graphics-and-animations-for-ios-apps/>
///
/// See also these links:
/// - <https://developer.apple.com/library/archive/documentation/Performance/Conceptual/OpenCL_MacProgGuide/SynchronizingIOSurfacesAcrossProcessors/SynchronizingIOSurfacesAcrossProcessors.html>
/// - <http://russbishop.net/cross-process-rendering>
/// - <https://www.chromium.org/developers/design-documents/iosurface-meeting-notes/>
/// - <https://github.com/gpuweb/gpuweb/issues/2535>
/// - <https://github.com/Me1000/out-of-process-calayer-rendering>
#[derive(Debug)]
struct Buffer {
    surface: CFRetained<IOSurfaceRef>,
    age: u8,
}

// SAFETY: `IOSurface` is marked `NS_SWIFT_SENDABLE`, and we only mutate it when we know it's not
// referenced by anything else (which we ensure by locking), and only then behind `&mut`.
unsafe impl Send for Buffer {}
// SAFETY: Same as above.
unsafe impl Sync for Buffer {}

impl Buffer {
    // The compositor shouldn't be writing to our surface, let's ensure that with this flag.
    const LOCK_OPTIONS: IOSurfaceLockOptions = IOSurfaceLockOptions::AvoidSync;

    /// Pixel format guaranteed to be supported by `CALayer.contents`; see `properties`.
    const PIXEL_FORMAT: u32 = kCVPixelFormatType_32BGRA;

    /// Bytes per pixel for `PIXEL_FORMAT`.
    const BYTES_PER_PIXEL: u32 = 4;

    fn new(width: u32, height: u32, color_space: &CGColorSpace) -> Self {
        // FIXME(madsmtm): Allow setting `write_combine_cache`:
        // https://github.com/rust-windowing/softbuffer/pull/320
        let properties = Self::properties(
            width,
            height,
            Self::PIXEL_FORMAT,
            Self::BYTES_PER_PIXEL,
            false,
        );
        let surface = unsafe { IOSurfaceRef::new(properties.as_opaque()) }.unwrap();
        let this = Self { surface, age: 0 };
        this.set_color_space(color_space);
        this
    }

    /// Get properties used when creating the buffer.
    ///
    /// NOTE: "Properties" are distinct from "values"; the former is immutable and can only be set
    /// upon creation, while the latter can be changed (with `IOSurfaceSetValue`).
    fn properties(
        width: u32,
        height: u32,
        pixel_format: u32,
        bytes_per_pixel: u32,
        write_combine_cache: bool,
    ) -> CFRetained<CFMutableDictionary<CFString, CFType>> {
        let properties = CFMutableDictionary::<CFString, CFType>::empty();

        // Set properties of the surface.
        properties.add(
            unsafe { kIOSurfaceWidth },
            &CFNumber::new_isize(width as isize),
        );
        properties.add(
            unsafe { kIOSurfaceHeight },
            &CFNumber::new_isize(height as isize),
        );
        // NOTE: If an unsupported pixel format is provided, the compositor usually won't render
        // anything (which means it'll render whatever was there before, very glitchy).
        //
        // The list of formats is hardware- and OS-dependent, see e.g. the following link:
        // https://developer.apple.com/forums/thread/673868
        //
        // Basically only `kCVPixelFormatType_32BGRA` is guaranteed to work, though from testing,
        // there's a few more that we might be able to use; see the following repository:
        // https://github.com/madsmtm/iosurface-calayer-formats
        properties.add(
            unsafe { kIOSurfacePixelFormat },
            &CFNumber::new_i32(pixel_format as i32),
        );
        properties.add(
            unsafe { kIOSurfaceBytesPerElement },
            &CFNumber::new_i32(bytes_per_pixel as i32),
        );

        // Be a bit more strict about usage of the surface in debug mode.
        #[cfg(debug_assertions)]
        properties.add(
            unsafe { objc2_io_surface::kIOSurfacePixelSizeCastingAllowed },
            &**objc2_core_foundation::CFBoolean::new(false),
        );

        if write_combine_cache {
            properties.add(
                unsafe { kIOSurfaceCacheMode },
                &**CFNumber::new_i32(kIOSurfaceMapWriteCombineCache as _),
            );
        }

        properties
    }

    /// Change the color space of the buffer.
    ///
    /// Defaults to the color space that the layer is currently on (so usually not what you want).
    fn set_color_space(&self, color_space: &CGColorSpace) {
        // This is a "value" we can change at runtime, not a "property" that is fixed at creation.
        unsafe {
            self.surface
                .set_value(kIOSurfaceColorSpace, &color_space.property_list().unwrap())
        }
    }

    #[track_caller]
    fn lock(&self) {
        let ret = unsafe { self.surface.lock(Self::LOCK_OPTIONS, ptr::null_mut()) };
        if ret != 0 {
            panic!("failed locking buffer: {ret}");
        }
    }

    #[track_caller]
    fn unlock(&self) {
        let ret = unsafe { self.surface.unlock(Self::LOCK_OPTIONS, ptr::null_mut()) };
        if ret != 0 {
            panic!("failed unlocking buffer: {ret}");
        }
    }

    /// # Safety
    ///
    /// The surface must be locked (done in `next_buffer`).
    unsafe fn data(&mut self) -> &mut [Pixel] {
        let num_bytes = self.surface.bytes_per_row() * self.surface.height();
        let ptr = self.surface.base_address().cast::<Pixel>();

        // SAFETY: `IOSurface` is a kernel-managed buffer, which means it's page-aligned, which is
        // plenty for the 4 byte alignment required here.
        //
        // Additionally, the buffer is owned by us, and we're the only ones that are going to write
        // to it. Since we re-use the buffer, it _might_ be read by the compositor while we write to
        // it - this is still sound on our side, though it might cause tearing. `next_buffer` waits
        // (bounded) on `is_in_use` to make that unlikely, but does not guarantee it, so writing here
        // while the compositor reads is possible and only risks visual tearing, not unsoundness.
        unsafe { slice::from_raw_parts_mut(ptr.as_ptr(), num_bytes / size_of::<Pixel>()) }
    }
}

#[derive(Debug)]
struct SendCALayer(Retained<CALayer>);

// SAFETY: CALayer is dubiously thread safe, like most things in Core Animation.
// But since we make sure to do our changes within a CATransaction, it is
// _probably_ fine for us to use CALayer from different threads.
//
// See also:
// https://developer.apple.com/documentation/quartzcore/catransaction/1448267-lock?language=objc
// https://stackoverflow.com/questions/76250226/how-to-render-content-of-calayer-on-a-background-thread
unsafe impl Send for SendCALayer {}
// SAFETY: Same as above.
unsafe impl Sync for SendCALayer {}

impl Deref for SendCALayer {
    type Target = CALayer;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Grabbed from `objc2-core-video` to avoid having to depend on that (for now at least).
#[allow(non_upper_case_globals)]
const kCVPixelFormatType_32BGRA: u32 = 0x42475241;
