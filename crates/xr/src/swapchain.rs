use std::sync::Arc;

use ambient_gpu::gpu::Gpu;
use ash::vk::{self, Handle};
use openxr as xr;

const WGPU_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const VK_COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

pub(crate) struct Swapchain {
    pub handle: xr::Swapchain<xr::Vulkan>,
    pub resolution: vk::Extent2D,
    buffers: Vec<wgpu::Texture>,
}

impl Swapchain {
    pub(crate) fn new(
        gpu: Arc<Gpu>,
        session: xr::Session<xr::Vulkan>,
        view: xr::ViewConfigurationView,
    ) -> Self {
        use wgpu_hal::{api::Vulkan as V, Api};
        let resolution = vk::Extent2D {
            width: view.recommended_image_rect_width,
            height: view.recommended_image_rect_height,
        };

        let handle = session
            .create_swapchain(&xr::SwapchainCreateInfo {
                create_flags: xr::SwapchainCreateFlags::EMPTY,
                usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                    | xr::SwapchainUsageFlags::SAMPLED,
                format: VK_COLOR_FORMAT.as_raw() as _,
                // The Vulkan graphics pipeline we create is not set up for multisampling,
                // so we hardcode this to 1. If we used a proper multisampling setup, we
                // could set this to `views[0].recommended_swapchain_sample_count`.
                sample_count: 1,
                width: resolution.width,
                height: resolution.height,
                face_count: 1,
                array_size: 2,
                mip_count: 1,
            })
            .unwrap();
        let images = handle.enumerate_images().unwrap();
        Self {
            handle,
            resolution,
            buffers: images
                .into_iter()
                .map(|color_image| {
                    let color_image = vk::Image::from_raw(color_image);
                    let wgpu_hal_texture = unsafe {
                        <V as Api>::Device::texture_from_raw(
                            color_image,
                            &wgpu_hal::TextureDescriptor {
                                label: Some("VR Swapchain"),
                                size: wgpu::Extent3d {
                                    width: resolution.width,
                                    height: resolution.height,
                                    depth_or_array_layers: 2,
                                },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: WGPU_COLOR_FORMAT,
                                usage: wgpu_hal::TextureUses::COLOR_TARGET
                                    | wgpu_hal::TextureUses::COPY_DST,
                                memory_flags: wgpu_hal::MemoryFlags::empty(),
                                view_formats: vec![],
                            },
                            None,
                        )
                    };
                    let texture = unsafe {
                        gpu.device.create_texture_from_hal::<V>(
                            wgpu_hal_texture,
                            &wgpu::TextureDescriptor {
                                label: Some("VR Swapchain"),
                                size: wgpu::Extent3d {
                                    width: resolution.width,
                                    height: resolution.height,
                                    depth_or_array_layers: 2,
                                },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: WGPU_COLOR_FORMAT,
                                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                                    | wgpu::TextureUsages::COPY_DST,
                                view_formats: &[],
                            },
                        )
                    };
                    texture
                })
                .collect(),
        }
    }

    pub(crate) fn get_render_view(&mut self) -> wgpu::TextureView {
        let image_index = self.handle.acquire_image().unwrap();
        self.handle.wait_image(xr::Duration::INFINITE).unwrap();

        let texture = &self.buffers[image_index as usize];

        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(2),
            ..Default::default()
        })
    }
}
