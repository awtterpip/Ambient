use std::sync::Arc;

use ambient_gpu::gpu::Gpu;
use ash::vk::{self, Handle};
use openxr as xr;

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
                format: wgpu_to_vulkan(gpu.swapchain_format()).as_raw() as _,
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
                                format: gpu.swapchain_format(),
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
                                format: gpu.swapchain_format(),
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

    pub(crate) fn get_render_view(&mut self, ) -> wgpu::TextureView {
        let image_index = self.handle.acquire_image().unwrap();
        self.handle.wait_image(xr::Duration::INFINITE).unwrap();

        let texture = &self.buffers[image_index as usize];

        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(2),
            ..Default::default()
        })
    }

    pub(crate) fn get_single_render_view(&mut self) -> wgpu::TextureView {
        let image_index = self.handle.acquire_image().unwrap();
        self.handle.wait_image(xr::Duration::INFINITE).unwrap();

        let texture = &self.buffers[image_index as usize];

        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            array_layer_count: Some(1),
            ..Default::default()
        })
    }
}

fn wgpu_to_vulkan(format: wgpu::TextureFormat) -> vk::Format {
    use vk::Format;
    match format {
        wgpu::TextureFormat::R8Unorm => Format::R8_UNORM,
        wgpu::TextureFormat::R8Snorm => Format::R8_SNORM,
        wgpu::TextureFormat::R8Uint => Format::R8_UINT,
        wgpu::TextureFormat::R8Sint => Format::R8_SINT,
        wgpu::TextureFormat::R16Uint => Format::R16_UINT,
        wgpu::TextureFormat::R16Sint => Format::R16_SINT,
        wgpu::TextureFormat::R16Unorm => Format::R16_UNORM,
        wgpu::TextureFormat::R16Snorm => Format::R16_SNORM,
        wgpu::TextureFormat::R16Float => Format::R16_SFLOAT,
        wgpu::TextureFormat::Rg8Unorm => Format::R8G8_UNORM,
        wgpu::TextureFormat::Rg8Snorm => Format::R8G8_SNORM,
        wgpu::TextureFormat::Rg8Uint => Format::R8G8_UINT,
        wgpu::TextureFormat::Rg8Sint => Format::R8G8_SINT,
        wgpu::TextureFormat::R32Uint => Format::R32_UINT,
        wgpu::TextureFormat::R32Sint => Format::R32_SINT,
        wgpu::TextureFormat::R32Float => Format::R32_SFLOAT,
        wgpu::TextureFormat::Rg16Uint => Format::R16G16_UINT,
        wgpu::TextureFormat::Rg16Sint => Format::R16G16_SINT,
        wgpu::TextureFormat::Rg16Unorm => Format::R16G16_UNORM,
        wgpu::TextureFormat::Rg16Snorm => Format::R16G16_SNORM,
        wgpu::TextureFormat::Rg16Float => Format::R16G16_SFLOAT,
        wgpu::TextureFormat::Rgba8Unorm => Format::R8G8B8A8_UNORM,
        wgpu::TextureFormat::Rgba8UnormSrgb => Format::R8G8B8A8_SRGB,
        wgpu::TextureFormat::Rgba8Snorm => Format::R8G8B8A8_SNORM,
        wgpu::TextureFormat::Rgba8Uint => Format::R8G8B8A8_UINT,
        wgpu::TextureFormat::Rgba8Sint => Format::R8G8B8A8_SINT,
        wgpu::TextureFormat::Bgra8Unorm => Format::B8G8R8A8_UNORM,
        wgpu::TextureFormat::Bgra8UnormSrgb => Format::B8G8R8A8_SRGB,
        wgpu::TextureFormat::Rgb9e5Ufloat => Format::E5B9G9R9_UFLOAT_PACK32, // this might be the wrong type??? i can't tell
        wgpu::TextureFormat::Rgb10a2Unorm => Format::A2R10G10B10_UNORM_PACK32,
        wgpu::TextureFormat::Rg11b10Float => panic!("this texture type invokes nothing but fear within my soul and i don't think vulkan has a proper type for this"),
        wgpu::TextureFormat::Rg32Uint => Format::R32G32_UINT,
        wgpu::TextureFormat::Rg32Sint => Format::R32G32_SINT,
        wgpu::TextureFormat::Rg32Float => Format::R32G32_SFLOAT,
        wgpu::TextureFormat::Rgba16Uint => Format::R16G16B16A16_UINT,
        wgpu::TextureFormat::Rgba16Sint => Format::R16G16B16A16_SINT,
        wgpu::TextureFormat::Rgba16Unorm => Format::R16G16B16A16_UNORM,
        wgpu::TextureFormat::Rgba16Snorm => Format::R16G16B16A16_SNORM,
        wgpu::TextureFormat::Rgba16Float => Format::R16G16B16A16_SFLOAT,
        wgpu::TextureFormat::Rgba32Uint => Format::R32G32B32A32_UINT,
        wgpu::TextureFormat::Rgba32Sint => Format::R32G32B32A32_SINT,
        wgpu::TextureFormat::Rgba32Float => Format::R32G32B32A32_SFLOAT,
        wgpu::TextureFormat::Stencil8 => Format::S8_UINT,
        wgpu::TextureFormat::Depth16Unorm => Format::D16_UNORM,
        wgpu::TextureFormat::Depth24Plus => Format::X8_D24_UNORM_PACK32,
        wgpu::TextureFormat::Depth24PlusStencil8 => Format::D24_UNORM_S8_UINT,
        wgpu::TextureFormat::Depth32Float => Format::D32_SFLOAT,
        wgpu::TextureFormat::Depth32FloatStencil8 => Format::D32_SFLOAT_S8_UINT,
        wgpu::TextureFormat::Etc2Rgb8Unorm => Format::ETC2_R8G8B8_UNORM_BLOCK,
        wgpu::TextureFormat::Etc2Rgb8UnormSrgb => Format::ETC2_R8G8B8_SRGB_BLOCK,
        wgpu::TextureFormat::Etc2Rgb8A1Unorm => Format::ETC2_R8G8B8A1_UNORM_BLOCK,
        wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb => Format::ETC2_R8G8B8A1_SRGB_BLOCK,
        wgpu::TextureFormat::Etc2Rgba8Unorm => Format::ETC2_R8G8B8A8_UNORM_BLOCK,
        wgpu::TextureFormat::Etc2Rgba8UnormSrgb => Format::ETC2_R8G8B8A8_SRGB_BLOCK,
        wgpu::TextureFormat::EacR11Unorm => Format::EAC_R11_UNORM_BLOCK,
        wgpu::TextureFormat::EacR11Snorm => Format::EAC_R11_SNORM_BLOCK,
        wgpu::TextureFormat::EacRg11Unorm => Format::EAC_R11G11_UNORM_BLOCK,
        wgpu::TextureFormat::EacRg11Snorm => Format::EAC_R11G11_SNORM_BLOCK,
        wgpu::TextureFormat::Astc { block, channel } => panic!("please god kill me now"),
        _ => panic!("fuck no")
    }
}