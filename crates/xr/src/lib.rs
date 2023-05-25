mod swapchain;
mod rendering;
mod input;

use std::{
    ffi::{c_void, CString},
    sync::{atomic::AtomicBool, Arc},
};

use ambient_gpu::{gpu::Gpu, settings::Settings};
use ambient_std::asset_cache::SyncAssetKey;
use anyhow::Context;
use ash::vk::{self, Handle};
use glam::{uvec2, Vec3, Quat};
use input::XrInput;
use openxr as xr;
use parking_lot::Mutex;
use rendering::FrameInner;
use winit::window::Window;

const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;

type XrPose = (Vec3, Quat);

pub struct PostFrameData {
    pub left_hand: Option<XrPose>,
    pub right_hand: Option<XrPose>,
    pub target: Option<wgpu::TextureView>,
    pub views: Vec<xr::View>,
}


#[derive(Debug, Clone)]
pub struct XrKey;
impl SyncAssetKey<Arc<XrState>> for XrKey {}

pub struct XrState {
    instance: xr::Instance,
    session: xr::Session<xr::Vulkan>,
    session_running: AtomicBool,
    frame: Mutex<FrameInner>,
    input: XrInput,
    event_buffer: xr::EventDataBuffer,
}

unsafe impl Sync for XrState {}
unsafe impl Send for XrState {}

impl XrState {
    #[tracing::instrument(level = "info")]
    pub async fn initialize_with_wgpu(
        window: Option<&Window>,
        will_be_polled: bool,
        settings: &Settings,
    ) -> anyhow::Result<(Arc<Gpu>, Arc<Self>)> {
        use wgpu_hal::{api::Vulkan as V, Api};

        let xr_entry = unsafe { xr::Entry::load() }?;

        let available_extensions = xr_entry.enumerate_extensions()?;
        assert!(available_extensions.khr_vulkan_enable2);
        tracing::info!("available xr exts: {:#?}", available_extensions);

        let mut enabled_extensions = xr::ExtensionSet::default();
        enabled_extensions.khr_vulkan_enable2 = true;
        #[cfg(target_os = "android")]
        {
            enabled_extensions.khr_android_create_instance = true;
        }

        let available_layers = xr_entry.enumerate_layers()?;
        tracing::info!("available xr layers: {:#?}", available_layers);

        let xr_instance = xr_entry.create_instance(
            &xr::ApplicationInfo {
                application_name: "Ambient",
                ..Default::default()
            },
            &enabled_extensions,
            &[],
        )?;
        tracing::info!("created instance");
        let instance_props = xr_instance.properties()?;
        let xr_system_id = xr_instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
        tracing::info!("created system");
        let system_props = xr_instance.system_properties(xr_system_id).unwrap();
        tracing::info!(
            "loaded OpenXR runtime: {} {} {}",
            instance_props.runtime_name,
            instance_props.runtime_version,
            if system_props.system_name.is_empty() {
                "<unnamed>"
            } else {
                &system_props.system_name
            }
        );

        let blend_mode =
            xr_instance.enumerate_environment_blend_modes(xr_system_id, VIEW_TYPE)?[0];

        let vk_target_version = vk::make_api_version(0, 1, 1, 0);
        let vk_target_version_xr = xr::Version::new(1, 1, 0);
        let reqs = xr_instance.graphics_requirements::<xr::Vulkan>(xr_system_id)?;
        if vk_target_version_xr < reqs.min_api_version_supported
            || vk_target_version_xr.major() > reqs.max_api_version_supported.major()
        {
            panic!(
                "OpenXR runtime requires Vulkan version > {}, < {}.0.0",
                reqs.min_api_version_supported,
                reqs.max_api_version_supported.major() + 1
            );
        }

        let vk_entry = unsafe { ash::Entry::load() }?;
        let flags = wgpu_hal::InstanceFlags::empty();
        let extensions =
            <V as Api>::Instance::required_extensions(&vk_entry, vk_target_version, flags)?;
        let device_extensions = vec![ash::extensions::khr::Swapchain::name(), ash::extensions::khr::DrawIndirectCount::name()];
        tracing::info!(
            "creating vulkan instance with these extensions: {:#?}",
            extensions
        );

        let vk_instance = unsafe {
            let extensions_cchar: Vec<_> = extensions.iter().map(|s| s.as_ptr()).collect();

            let app_name = CString::new("Ambient")?;
            let vk_app_info = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(1)
                .engine_name(&app_name)
                .engine_version(1)
                .api_version(vk_target_version);

            let vk_instance = xr_instance
                .create_vulkan_instance(
                    xr_system_id,
                    std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                    &vk::InstanceCreateInfo::builder()
                        .application_info(&vk_app_info)
                        .enabled_extension_names(&extensions_cchar) as *const _
                        as *const _,
                )
                .context("XR error creating Vulkan instance")
                .unwrap()
                .map_err(vk::Result::from_raw)
                .context("Vulkan error creating Vulkan instance")
                .unwrap();

            ash::Instance::load(
                vk_entry.static_fn(),
                vk::Instance::from_raw(vk_instance as _),
            )
        };
        tracing::info!("created vulkan instance");

        let vk_instance_ptr = vk_instance.handle().as_raw() as *const c_void;

        let vk_physical_device = vk::PhysicalDevice::from_raw(unsafe {
            xr_instance.vulkan_graphics_device(xr_system_id, vk_instance.handle().as_raw() as _)?
                as _
        });
        let vk_physical_device_ptr = vk_physical_device.as_raw() as *const c_void;

        let vk_device_properties =
            unsafe { vk_instance.get_physical_device_properties(vk_physical_device) };
        if vk_device_properties.api_version < vk_target_version {
            unsafe { vk_instance.destroy_instance(None) }
            panic!("Vulkan physical device doesn't support version 1.1");
        }

        let wgpu_vk_instance = unsafe {
            <V as Api>::Instance::from_raw(
                vk_entry.clone(),
                vk_instance.clone(),
                vk_target_version,
                0,
                extensions,
                flags,
                false,
                Some(Box::new(())),
            )?
        };

        let wgpu_features =
            wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | wgpu::Features::MULTIVIEW | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | wgpu::Features::MULTI_DRAW_INDIRECT;

        let wgpu_exposed_adapter = wgpu_vk_instance
            .expose_adapter(vk_physical_device)
            .context("failed to expose adapter")?;

        let enabled_extensions = wgpu_exposed_adapter
            .adapter
            .required_device_extensions(wgpu_features);

        let (wgpu_open_device, vk_device_ptr, queue_family_index) = {
            let extensions_cchar: Vec<_> = device_extensions.iter().map(|s| s.as_ptr()).collect();
            let mut enabled_phd_features = wgpu_exposed_adapter
                .adapter
                .physical_device_features(&enabled_extensions, wgpu_features);
            let family_index = 0;
            let family_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(family_index)
                .queue_priorities(&[1.0])
                .build();
            let family_infos = [family_info];
            let info = enabled_phd_features
                .add_to_device_create_builder(
                    vk::DeviceCreateInfo::builder()
                        .queue_create_infos(&family_infos)
                        .push_next(&mut vk::PhysicalDeviceMultiviewFeatures {
                            multiview: vk::TRUE,
                            ..Default::default()
                        }),
                )
                .enabled_extension_names(&extensions_cchar)
                .build();
            let vk_device = unsafe {
                let vk_device = xr_instance
                    .create_vulkan_device(
                        xr_system_id,
                        std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                        vk_physical_device.as_raw() as _,
                        &info as *const _ as *const _,
                    )
                    .context("XR error creating Vulkan device")?
                    .map_err(vk::Result::from_raw)
                    .context("Vulkan error creating Vulkan device")?;

                ash::Device::load(vk_instance.fp_v1_0(), vk::Device::from_raw(vk_device as _))
            };
            let vk_device_ptr = vk_device.handle().as_raw() as *const c_void;

            let wgpu_open_device = unsafe {
                wgpu_exposed_adapter.adapter.device_from_raw(
                    vk_device,
                    true,
                    &enabled_extensions,
                    wgpu_features,
                    family_info.queue_family_index,
                    0,
                )
            }?;

            (
                wgpu_open_device,
                vk_device_ptr,
                family_info.queue_family_index,
            )
        };

        let wgpu_instance =
            unsafe { wgpu::Instance::from_hal::<wgpu_hal::api::Vulkan>(wgpu_vk_instance) };
        let wgpu_adapter = unsafe { wgpu_instance.create_adapter_from_hal(wgpu_exposed_adapter) };
        let (wgpu_device, wgpu_queue) = unsafe {
            wgpu_adapter.create_device_from_hal(
                wgpu_open_device,
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu_features,
                    limits: wgpu::Limits {
                        max_bind_groups: 8,
                        max_storage_buffer_binding_size: wgpu_adapter
                            .limits()
                            .max_storage_buffer_binding_size,
                        max_push_constant_size: 4,
                        ..Default::default()
                    },
                },
                None,
            )
        }?;

        let (session, frame_wait, frame_stream) = unsafe {
            xr_instance.create_session::<xr::Vulkan>(
                xr_system_id,
                &xr::vulkan::SessionCreateInfo {
                    instance: vk_instance_ptr,
                    physical_device: vk_physical_device_ptr,
                    device: vk_device_ptr,
                    queue_family_index,
                    queue_index: 0,
                },
            )
        }?;

        let surface = window.map(|window| unsafe { wgpu_instance.create_surface(window).unwrap() });
        let swapchain_format = surface
            .as_ref()
            .map(|surface| surface.get_capabilities(&wgpu_adapter).formats[0]);
        tracing::info!("Swapchain format: {swapchain_format:?}");
        let swapchain_mode = if surface.is_some() {
            if settings.vsync() {
                // From wgpu docs:
                // "Chooses FifoRelaxed -> Fifo based on availability."
                Some(wgpu::PresentMode::AutoVsync)
            } else {
                // From wgpu docs:
                // "Chooses Immediate -> Mailbox -> Fifo (on web) based on availability."
                Some(wgpu::PresentMode::AutoNoVsync)
            }
        } else {
            None
        };
        tracing::debug!("Swapchain present mode: {swapchain_mode:?}");

        if let (Some(window), Some(surface), Some(mode), Some(format)) =
            (window, &surface, swapchain_mode, swapchain_format)
        {
            let size = window.inner_size();
            surface.configure(
                &wgpu_device,
                &Gpu::create_sc_desc(format, mode, uvec2(size.width, size.height)),
            );
        }

        let views = xr_instance.enumerate_view_configuration_views(xr_system_id, VIEW_TYPE)?;

        let gpu = Arc::new(Gpu {
            surface,
            device: wgpu_device,
            queue: wgpu_queue,
            swapchain_format,
            swapchain_mode,
            will_be_polled,
            adapter: wgpu_adapter,
        });

        let xr_state = Arc::new(Self {
            input: XrInput::new(xr_instance.clone(), session.clone())?,
            instance: xr_instance,
            session: session.clone(),
            session_running: AtomicBool::new(false),
            frame: Mutex::new(FrameInner {
                session,
                gpu: gpu.clone(),
                blend_mode,
                wait: frame_wait,
                stream: frame_stream,
                swapchain: None,
                views,
            }),
            event_buffer: xr::EventDataBuffer::new(),
        });

        Ok((gpu, xr_state))
    }

    pub fn pre_frame(&self) -> anyhow::Result<Option<xr::FrameState>> {
        let event_buffer = unsafe {
            &mut *(&self.event_buffer as *const _ as *mut _)
        };
        while let Some(event) = self.instance.poll_event(event_buffer)? {
            use xr::Event::*;
            match event {
                SessionStateChanged(e) => {
                    // Session state change is where we can begin and end sessions, as well as
                    // find quit messages!
                    tracing::info!("entered state {:?}", e.state());
                    match e.state() {
                        xr::SessionState::READY => {
                            self.session.begin(VIEW_TYPE)?;
                            self.session_running.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        xr::SessionState::STOPPING => {
                            self.session.end()?;
                            self.session_running.store(false, std::sync::atomic::Ordering::Relaxed);
                        }
                        xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                            return Ok(None);
                        }
                        _ => {}
                    }
                }
                InstanceLossPending(_) => {
                    return Ok(None);
                }
                EventsLost(e) => {
                    tracing::warn!("lost {} events", e.lost_event_count());
                }
                _ => {}
            }
        }
        if !self.session_running.load(std::sync::atomic::Ordering::Relaxed) {
            // Don't grind up the CPU
            std::thread::sleep(std::time::Duration::from_millis(10));
            return Ok(None);
        }

        // Block until the previous frame is finished displaying, and is ready for another one.
        // Also returns a prediction of when the next frame will be displayed, for use with
        // predicting locations of controllers, viewpoints, etc.
        let xr_frame_state = self.frame.lock().begin()?;

        Ok(Some(xr_frame_state))
    }

    pub fn post_frame(&self, xr_frame_state: xr::FrameState) -> anyhow::Result<PostFrameData> {
        let mut pfd = self.input.post_frame(xr_frame_state)?;
        pfd.target = Some(self.frame.lock().get_single_render_view());
        Ok(pfd)
    }

    pub fn post_queue_submit(
        &self,
        xr_frame_state: xr::FrameState,
        views: &[openxr::View],
    ) -> anyhow::Result<()> {
        self.frame.lock().post_queue_submit(xr_frame_state, views, self.input.stage())
    }

    pub fn left_hand(&self) -> XrPose {
        self.input.left_hand()
    }

    pub fn right_hand(&self) -> XrPose {
        self.input.right_hand()
    }

    pub fn views(&self) -> Vec<XrPose> {
        self.input.views()
    }
}

