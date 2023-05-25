use std::sync::Arc;

use crate::swapchain::Swapchain;
use ambient_gpu::gpu::Gpu;
use openxr as xr;

pub(crate) struct FrameInner {
    pub gpu: Arc<Gpu>,
    pub session: xr::Session<xr::Vulkan>,
    pub wait: xr::FrameWaiter,
    pub stream: xr::FrameStream<xr::Vulkan>,
    pub blend_mode: xr::EnvironmentBlendMode,
    pub swapchain: Option<Swapchain>,
    pub views: Vec<xr::ViewConfigurationView>,
}

impl FrameInner {
    pub fn begin(&mut self) -> anyhow::Result<xr::FrameState> {
        let frame_state = self.wait.wait()?;
        self.stream.begin()?;
        Ok(frame_state)
    }

    pub fn get_render_view(&mut self) -> wgpu::TextureView {
        let swapchain = self.swapchain.get_or_insert_with(|| {
            Swapchain::new(self.gpu.clone(), self.session.clone(), self.views[0])
        });

        swapchain.get_render_view()
    }

    pub fn get_single_render_view(&mut self) -> wgpu::TextureView {
        let swapchain = self.swapchain.get_or_insert_with(|| {
            Swapchain::new(self.gpu.clone(), self.session.clone(), self.views[0])
        });

        swapchain.get_single_render_view()
    }

    pub fn post_queue_submit(
        &mut self,
        xr_frame_state: xr::FrameState,
        views: &[openxr::View],
        stage: &xr::Space,
    ) -> anyhow::Result<()> {
        if let Some(swapchain) = &mut self.swapchain {
            swapchain.handle.release_image()?;
            let rect = xr::Rect2Di {
                offset: xr::Offset2Di { x: 0, y: 0 },
                extent: xr::Extent2Di {
                    width: swapchain.resolution.width as _,
                    height: swapchain.resolution.height as _,
                },
            };
            self.stream.end(
                xr_frame_state.predicted_display_time,
                self.blend_mode,
                &[&xr::CompositionLayerProjection::new().space(stage).views(&[
                    xr::CompositionLayerProjectionView::new()
                        .pose(views[0].pose)
                        .fov(views[0].fov)
                        .sub_image(
                            xr::SwapchainSubImage::new()
                                .swapchain(&swapchain.handle)
                                .image_array_index(0)
                                .image_rect(rect),
                        ),
                    xr::CompositionLayerProjectionView::new()
                        .pose(views[1].pose)
                        .fov(views[1].fov)
                        .sub_image(
                            xr::SwapchainSubImage::new()
                                .swapchain(&swapchain.handle)
                                .image_array_index(1)
                                .image_rect(rect),
                        ),
                ])],
            )?;
        }

        Ok(())
    }
}
