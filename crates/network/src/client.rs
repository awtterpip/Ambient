use ambient_core::{gpu, window::window_scale_factor};
use ambient_ecs::{components, ComponentValueBase, Resource, World};
use ambient_element::{element_component, Element, ElementComponentExt, Hooks};
use ambient_renderer::RenderTarget;
use ambient_rpc::RpcRegistry;
use ambient_std::{asset_cache::AssetCache, cb, friendly_id, to_byte_unit, Cb};
use ambient_ui_native::{Image, MeasureSize};
use bytes::{BufMut, Bytes, BytesMut};
use futures::future::BoxFuture;
use glam::UVec2;
use parking_lot::Mutex;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    future::Future,
    pin::Pin,
    sync::Arc,
};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};

use crate::{
    client_connection::ConnectionKind, client_game_state::ClientGameState, log_network_result,
    proto::client::SharedClientState, server, NetworkError, MAX_FRAME_SIZE, RPC_BISTREAM_ID,
};

components!("network::client", {
    @[Resource]
    game_client: Option<GameClient>,
    @[Resource]
    bi_stream_handlers: BiStreamHandlers,
    @[Resource]
    uni_stream_handlers: UniStreamHandlers,
    @[Resource]
    datagram_handlers: DatagramHandlers,
    /// The most recent server performance statistics
    @[Resource]
    client_network_stats: NetworkStats,
});

pub type DynSend = Pin<Box<dyn AsyncWrite + Send + Sync>>;
pub type DynRecv = Pin<Box<dyn AsyncRead + Send + Sync>>;

type BiStreamHandler = Arc<dyn Fn(&mut World, AssetCache, DynSend, DynRecv) + Sync + Send>;
type UniStreamHandler = Arc<dyn Fn(&mut World, AssetCache, DynRecv) + Sync + Send>;
type DatagramHandler = Arc<dyn Fn(&mut World, AssetCache, Bytes) + Sync + Send>;

pub type BiStreamHandlers = HashMap<u32, (&'static str, BiStreamHandler)>;
pub type UniStreamHandlers = HashMap<u32, (&'static str, UniStreamHandler)>;
pub type DatagramHandlers = HashMap<u32, (&'static str, DatagramHandler)>;

/// Represents either side of a high level connection to a game client of some sort.
///
/// Allows making requests and RPC, etc
pub trait ClientConnection: 'static + Send + Sync {
    /// Performs a bidirectional request and waits for a response.
    fn request_bi(&self, id: u32, data: Bytes) -> BoxFuture<Result<Bytes, NetworkError>>;
    /// Performs a unidirectional request without waiting for a response.
    fn request_uni(&self, id: u32, data: Bytes) -> BoxFuture<Result<(), NetworkError>>;
    fn send_datagram(&self, id: u32, data: Bytes) -> Result<(), NetworkError>;
}

impl ClientConnection for quinn::Connection {
    fn request_bi(&self, id: u32, data: Bytes) -> BoxFuture<Result<Bytes, NetworkError>> {
        Box::pin(async move {
            let (mut send, recv) = self.open_bi().await?;

            send.write_u32(id).await?;
            send.write_all(&data).await?;

            drop(send);

            let buf = recv.read_to_end(MAX_FRAME_SIZE).await?.into();

            Ok(buf)
        })
    }

    fn request_uni(&self, id: u32, data: Bytes) -> BoxFuture<Result<(), NetworkError>> {
        Box::pin(async move {
            let mut send = self.open_uni().await?;

            send.write_u32(id).await?;
            send.write_all(&data).await?;

            Ok(())
        })
    }

    fn send_datagram(&self, id: u32, data: Bytes) -> Result<(), NetworkError> {
        let mut bytes = BytesMut::with_capacity(4 + data.len());
        bytes.put_u32(id);
        bytes.put(data);

        self.send_datagram(bytes.freeze())?;

        Ok(())
    }
}

impl ClientConnection for ConnectionKind {
    fn request_bi(&self, id: u32, data: Bytes) -> BoxFuture<Result<Bytes, NetworkError>> {
        Box::pin(async move {
            let (mut send, recv) = self.open_bi().await?;

            send.write_u32(id).await?;
            send.write_all(&data).await?;

            drop(send);

            let buf = recv.read_to_end(MAX_FRAME_SIZE).await?.into();

            Ok(buf)
        })
    }

    fn request_uni(&self, id: u32, data: Bytes) -> BoxFuture<Result<(), NetworkError>> {
        Box::pin(async move {
            let mut send = self.open_uni().await?;

            send.write_u32(id).await?;
            send.write_all(&data).await?;

            Ok(())
        })
    }

    fn send_datagram(&self, id: u32, data: Bytes) -> Result<(), NetworkError> {
        let mut bytes = BytesMut::with_capacity(4 + data.len());
        bytes.put_u32(id);
        bytes.put(data);

        self.send_datagram(bytes.freeze())?;

        Ok(())
    }
}

#[derive(Clone)]
/// Manages the client side connection to the server.
pub struct GameClient {
    pub connection: Arc<dyn ClientConnection>,
    pub rpc_registry: Arc<RpcRegistry<server::RpcArgs>>,
    pub user_id: String,
    pub game_state: SharedClientState,
    pub uid: String,
}

impl Debug for GameClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GameClient")
            .field("connection", &self.connection.type_name())
            .field("rpc_registry", &self.rpc_registry)
            .field("user_id", &self.user_id)
            .field("game_state", &self.game_state)
            .field("uid", &self.uid)
            .finish()
    }
}

impl GameClient {
    pub fn new(
        connection: Arc<dyn ClientConnection>,
        rpc_registry: Arc<RpcRegistry<server::RpcArgs>>,
        game_state: Arc<Mutex<ClientGameState>>,
        user_id: String,
    ) -> Self {
        Self {
            connection,
            rpc_registry,
            user_id,
            game_state,
            uid: friendly_id(),
        }
    }

    pub async fn rpc<
        Req: Serialize + DeserializeOwned + Send + 'static,
        Resp: Serialize + DeserializeOwned + Send,
        F: Fn(server::RpcArgs, Req) -> L + Send + Sync + Copy + 'static,
        L: Future<Output = Resp> + Send,
    >(
        &self,
        func: F,
        req: Req,
    ) -> Result<Resp, NetworkError> {
        rpc_request(&*self.connection, self.rpc_registry.clone(), func, req).await
    }

    pub fn make_standalone_rpc_wrapper<
        Req: Serialize + DeserializeOwned + Send + 'static,
        Resp: Serialize + DeserializeOwned + Send,
        F: Fn(server::RpcArgs, Req) -> L + Send + Sync + Copy + 'static,
        L: Future<Output = Resp> + Send,
    >(
        &self,
        runtime: &tokio::runtime::Handle,
        func: F,
    ) -> Cb<impl Fn(Req)> {
        let runtime = runtime.clone();
        let (connection, rpc_registry) = (self.connection.clone(), self.rpc_registry.clone());
        cb(move |req| {
            let (connection, rpc_registry) = (connection.clone(), rpc_registry.clone());
            runtime.spawn(async move {
                log_network_result!(rpc_request(&*connection, rpc_registry, func, req).await);
            });
        })
    }

    pub fn with_physics_world<R>(&self, f: impl Fn(&mut World) -> R) -> R {
        f(&mut self.game_state.lock().world)
    }
}

async fn rpc_request<
    Args: Send + 'static,
    Req: Serialize + DeserializeOwned + Send + 'static,
    Resp: Serialize + DeserializeOwned + Send,
    F: Fn(Args, Req) -> L + Send + Sync + Copy + 'static,
    L: Future<Output = Resp> + Send,
>(
    conn: &dyn ClientConnection,
    reg: Arc<RpcRegistry<Args>>,
    func: F,
    req: Req,
) -> Result<Resp, NetworkError> {
    let req = reg.serialize_req(func, req);

    let resp = conn.request_bi(RPC_BISTREAM_ID, req.into()).await?;

    let resp = reg.deserialize_resp(func, &resp)?;
    Ok(resp)
}

#[derive(Debug, Clone)]
pub struct GameClientRenderTarget(pub Arc<RenderTarget>);

#[derive(Debug)]
pub struct UseOnce<T> {
    val: Mutex<Option<T>>,
}

impl<T> UseOnce<T> {
    pub fn new(val: T) -> Self {
        Self {
            val: Mutex::new(Some(val)),
        }
    }

    pub fn take(&self) -> Option<T> {
        self.val.lock().take()
    }
}

pub type CleanupFunc = Box<dyn FnOnce() + Send + Sync>;
pub type LoadedFunc = Cb<dyn Fn(GameClient) -> anyhow::Result<CleanupFunc> + Send + Sync>;

#[element_component]
pub fn GameClientWorld(hooks: &mut Hooks) -> Element {
    let (render_target, set_render_target) =
        hooks.consume_context::<GameClientRenderTarget>().unwrap();
    let gpu = hooks.world.resource(gpu()).clone();
    let scale_factor = *hooks.world.resource(window_scale_factor());
    MeasureSize::el(
        Image {
            texture: Some(Arc::new(
                render_target
                    .0
                    .color_buffer
                    .create_view(&Default::default()),
            )),
        }
        .el(),
        cb(move |size| {
            set_render_target(GameClientRenderTarget(Arc::new(RenderTarget::new(
                gpu.clone(),
                (size * scale_factor as f32).as_uvec2().max(UVec2::ONE),
                None,
            ))))
        }),
    )
}

/// Set up and manage a connection to the server
#[derive(Debug, Clone, Default)]
pub struct NetworkStats {
    pub latency_ms: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

impl Display for NetworkStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} ms rtt, {}/s out, {}/s in",
            self.latency_ms,
            to_byte_unit(self.bytes_sent),
            to_byte_unit(self.bytes_received)
        )
    }
}
