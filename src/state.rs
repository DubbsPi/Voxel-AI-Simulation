use glam::{
    IVec2, Mat4, Vec3
};
use std::{
    sync::Arc,
    time::Instant,
    io::{self, Write},
    thread
};
use crossbeam_channel::{unbounded, Sender, Receiver};
use wgpu::util::DeviceExt;
use winit::keyboard::KeyCode;
use std::collections::{HashSet, HashMap};
use winit::{dpi::PhysicalSize, window::Window};

use crate::worldgen;
use worldgen::{BlockType, WorldGen};


pub const CHUNK_SIZE: i32 = 16;
pub const Y_DIST: IVec2 = IVec2::new(-6, 6);
pub const RENDER_DISTANCE: i32 = 8;
pub const WORLD_SEED: i32 = 1337;
pub const CHUNK_THREADS: i16 = 8;

pub const PADDED_SIZE: usize = (CHUNK_SIZE + 2) as usize;

const EST_VERTS_PER_CHUNK: usize = 10000;
const EST_INDICES_PER_CHUNK: usize = 15000;
const MAX_CHUNKS: usize = ((RENDER_DISTANCE * 2 + 2).pow(3)) as usize;


// Vertex uniform
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3]
}

impl Vertex {
    // Memory layout for the vertex buffer
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            // How many bytes per vertex
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // location 0 = position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // location 1 = normal
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // location 2 = color
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                }
            ]
        }
    }
}


// Camera uniform
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4]
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}


// Chunk uniform
pub struct Chunk {
    pub blocks: Box<[[[BlockType; PADDED_SIZE]; PADDED_SIZE]; PADDED_SIZE]>,

    // In chunk coords
    pub position: glam::IVec3,

    // GPU buffers
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,

    pub dirty: bool
}

impl Chunk {
    pub fn new(position: glam::IVec3) -> Self {
        Self {
            blocks: Box::new([[[BlockType::Air; PADDED_SIZE]; PADDED_SIZE]; PADDED_SIZE]),
            position,
            vertices: Vec::new(),
            indices: Vec::new(),
            dirty: true
        }
    }

    pub fn set_block(&mut self, x: usize, y: usize, z: usize, block: BlockType) {
        self.blocks[x + 1][y + 1][z + 1] = block; // Offset for padding
        self.dirty = true;
    }

    fn get_block(&self, x: i32, y: i32, z: i32, chunks: &HashMap<glam::IVec3, Chunk>) -> BlockType {
        if x >= 0 && x < CHUNK_SIZE && y >= 0 && y < CHUNK_SIZE && z >= 0 && z < CHUNK_SIZE {
            return self.blocks[x as usize][y as usize][z as usize];
        }

        let mut target_chunk = self.position;
        let mut lx = x;
        let mut ly = y;
        let mut lz = z;

        if lx < 0 { target_chunk.x -= 1; lx += CHUNK_SIZE; }
        else if lx >= CHUNK_SIZE { target_chunk.x += 1; lx -= CHUNK_SIZE; }

        if ly < 0 { target_chunk.y -= 1; ly += CHUNK_SIZE; }
        else if ly >= CHUNK_SIZE { target_chunk.y += 1; ly -= CHUNK_SIZE; }

        if lz < 0 { target_chunk.z -= 1; lz += CHUNK_SIZE; }
        else if lz >= CHUNK_SIZE { target_chunk.z += 1; lz -= CHUNK_SIZE; }

        if let Some(neighbor) = chunks.get(&target_chunk) {
            neighbor.blocks[lx as usize][ly as usize][lz as usize]
        } else {
            BlockType::Air
        }
    }

    pub fn rebuild_mesh(&mut self) {
        self.vertices.clear();
        self.indices.clear();

        let chunk_world = glam::Vec3::new(
            (self.position.x * CHUNK_SIZE) as f32,
            (self.position.y * CHUNK_SIZE) as f32,
            (self.position.z * CHUNK_SIZE) as f32,
        );

        // Ignore padded outsides
        for x in 1..=CHUNK_SIZE as usize {
            for y in 1..=CHUNK_SIZE as usize {
                for z in 1..=CHUNK_SIZE as usize {
                    if self.blocks[x][y][z] == BlockType::Air {
                        continue;
                    }

                    let world_pos = chunk_world
                        + glam::Vec3::new((x - 1) as f32, (y - 1) as f32, (z - 1) as f32);

                    if self.blocks[x + 1][y][z] == BlockType::Air {
                        add_face(&mut self.vertices, &mut self.indices, world_pos, Face::Right);
                    }
                    if self.blocks[x - 1][y][z] == BlockType::Air {
                        add_face(&mut self.vertices, &mut self.indices, world_pos, Face::Left);
                    }
                    if self.blocks[x][y + 1][z] == BlockType::Air {
                        add_face(&mut self.vertices, &mut self.indices, world_pos, Face::Top);
                    }
                    if self.blocks[x][y - 1][z] == BlockType::Air {
                        add_face(&mut self.vertices, &mut self.indices, world_pos, Face::Bottom);
                    }
                    if self.blocks[x][y][z + 1] == BlockType::Air {
                        add_face(&mut self.vertices, &mut self.indices, world_pos, Face::Front);
                    }
                    if self.blocks[x][y][z - 1] == BlockType::Air {
                        add_face(&mut self.vertices, &mut self.indices, world_pos, Face::Back);
                    }
                }
            }
        }

        self.dirty = false;
    }
}


#[derive(Clone, Copy)]
enum Face { Top, Bottom, Left, Right, Front, Back }

fn add_face(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, pos: glam::Vec3, face: Face) {
    let base = vertices.len() as u32;

    let (corners, normal, color) = match face {
        Face::Top => (
            [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
            [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]
        ),
        Face::Bottom => (
            [[0.0, 0.0, 0.0],[1.0, 0.0, 0.0 ], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [0.0, -1.0, 0.0], [1.0, 1.0, 1.0]
        ),
        Face::Right => (
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
            [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        ),
        Face::Left => (
            [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        ),
        Face::Front => (
            [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]
        ),
        Face::Back => (
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]
        )
    };

    for corner in corners {
        vertices.push(Vertex {
            position: [pos.x + corner[0], pos.y + corner[1], pos.z + corner[2]],
            normal,
            color,
        });
    }

    indices.extend_from_slice(&[
        base, base+1, base+2,
        base, base+2, base+3,
    ]);
}


// Threading for chunk generation
pub enum ChunkRequest {
    Load(glam::IVec3)
}
pub enum ChunkResponse {
    Loaded(glam::IVec3, Chunk)
}


// States of the engine
pub struct State {
    // Core objects
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    // Rendering
    render_pipeline: wgpu::RenderPipeline,
    
    // World generation and chunks
    chunks: HashMap<glam::IVec3, Chunk>,

    tx_request: Sender<ChunkRequest>,
    rx_response: Receiver<ChunkResponse>,
    pending_chunks: HashSet<glam::IVec3>,

    super_vertex_buffer: wgpu::Buffer,
    super_index_buffer: wgpu::Buffer,
    super_num_indices: u32,

    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    sampler: wgpu::Sampler,

    // Camera
    camera_buffer: wgpu::Buffer,
    camera_uniform: CameraUniform,
    camera_bind_group: wgpu::BindGroup,

    camera_pos: Vec3,
    camera_yaw: f32,
    camera_pitch: f32,
    camera_v: Vec3,

    mouse_grabbed: bool,
    held_keys: HashSet<KeyCode>,
    last_frame: Instant,
    start_time: Instant,

    // Background color
    clear_color: wgpu::Color
}

impl State {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // GPU connection
        let surface = instance
            .create_surface(window)
            .expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable gpu found");

        log::info!("Using adapter: {:?}", adapter.get_info().name);
        log::info!("Backend: {:?}", adapter.get_info().backend);

        // GPU device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("main device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_buffer_size: adapter.limits().max_buffer_size,
                        max_bind_groups: adapter.limits().max_bind_groups,
                        max_color_attachments: adapter.limits().max_color_attachments,
                        max_color_attachment_bytes_per_sample: adapter.limits().max_color_attachment_bytes_per_sample,
                        max_bindings_per_bind_group: adapter.limits().max_bindings_per_bind_group,
                        max_compute_invocations_per_workgroup: adapter.limits().max_compute_invocations_per_workgroup,
                        max_compute_workgroup_size_x: adapter.limits().max_compute_workgroup_size_x,
                        max_compute_workgroup_size_y: adapter.limits().max_compute_workgroup_size_y,
                        max_compute_workgroup_size_z: adapter.limits().max_compute_workgroup_size_z,
                        max_compute_workgroups_per_dimension: adapter.limits().max_compute_workgroups_per_dimension,
                        max_compute_workgroup_storage_size: adapter.limits().max_compute_workgroup_storage_size,
                        max_inter_stage_shader_components: adapter.limits().max_inter_stage_shader_components,
                        max_dynamic_uniform_buffers_per_pipeline_layout: adapter.limits().max_dynamic_uniform_buffers_per_pipeline_layout,
                        max_non_sampler_bindings: adapter.limits().max_non_sampler_bindings,
                        max_push_constant_size: adapter.limits().max_push_constant_size,
                        max_sampled_textures_per_shader_stage: adapter.limits().max_sampled_textures_per_shader_stage,
                        max_samplers_per_shader_stage: adapter.limits().max_samplers_per_shader_stage,
                        max_storage_buffers_per_shader_stage: adapter.limits().max_storage_buffers_per_shader_stage,
                        max_storage_textures_per_shader_stage: adapter.limits().max_storage_textures_per_shader_stage,
                        max_subgroup_size: adapter.limits().max_subgroup_size,
                        max_texture_array_layers: adapter.limits().max_texture_array_layers,
                        max_texture_dimension_1d: adapter.limits().max_texture_dimension_1d,
                        max_texture_dimension_2d: adapter.limits().max_texture_dimension_2d,
                        max_texture_dimension_3d: adapter.limits().max_texture_dimension_3d,
                        max_uniform_buffer_binding_size: adapter.limits().max_uniform_buffer_binding_size,
                        max_uniform_buffers_per_shader_stage: adapter.limits().max_uniform_buffers_per_shader_stage,
                        max_vertex_attributes: adapter.limits().max_vertex_attributes,
                        max_vertex_buffer_array_stride: adapter.limits().max_vertex_buffer_array_stride,
                        max_vertex_buffers: adapter.limits().max_vertex_buffers,
                        min_storage_buffer_offset_alignment: adapter.limits().min_storage_buffer_offset_alignment,
                        min_subgroup_size: adapter.limits().min_subgroup_size,
                        min_uniform_buffer_offset_alignment: adapter.limits().min_uniform_buffer_offset_alignment,
                        max_dynamic_storage_buffers_per_pipeline_layout: adapter.limits().max_dynamic_storage_buffers_per_pipeline_layout,
                        max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size
                    },
                    memory_hints: Default::default(),
                },
                None
            )
            .await
            .expect("failed to create device");


        // Swap chain config
        let surface_caps = surface.get_capabilities(&adapter);

        // Prefer sRGB
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        
        // WGSL shader
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        

        // Camera buffer
        let camera_uniform = CameraUniform::new();

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("camera bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            }
        );

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        
        // Threading setup
        let (tx_request, rx_request) = unbounded::<ChunkRequest>();
        let (tx_response, rx_response) = unbounded::<ChunkResponse>();

        let world_gen_arc = Arc::new(WorldGen::new(WORLD_SEED));

        for _ in 0..CHUNK_THREADS {
            let rx_req = rx_request.clone();
            let tx_res = tx_response.clone();
            let generation = Arc::clone(&world_gen_arc);

            thread::spawn(move || {
                while let Ok(request) = rx_req.recv() {
                    match request {
                        ChunkRequest::Load(pos) => {
                            let mut chunk = Chunk::new(pos);
                            generation.generate_chunk(&mut chunk);

                            chunk.rebuild_mesh();

                            let _ = tx_res.send(ChunkResponse::Loaded(pos, chunk));
                        }
                    }
                }
            });
        }

        // Super buffers
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Super Vertex Buffer"),
            size: (MAX_CHUNKS * EST_VERTS_PER_CHUNK * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Super Index Buffer"),
            size: (MAX_CHUNKS * EST_INDICES_PER_CHUNK * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pipeline layout
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render pipeline layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
        

        // Chunks
        let chunks = HashMap::new();

        let world_gen = WorldGen::new(WORLD_SEED);
        let y_pos = world_gen.surface_height(0, 0) as f32 + 2.0;

        // Depth texture
        let (depth_texture, depth_view, sampler) = Self::create_depth_texture(&device, &config);
                
        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,

            chunks,

            tx_request,
            rx_response,
            pending_chunks: HashSet::new(),

            super_vertex_buffer: vertex_buffer,
            super_index_buffer: index_buffer,
            super_num_indices: 0,

            depth_texture,
            depth_view,
            sampler,

            camera_buffer,
            camera_uniform,
            camera_bind_group,

            camera_pos: Vec3::new(0.0, y_pos, 0.0),
            camera_yaw: -std::f32::consts::FRAC_PI_2,
            camera_pitch: 0.0,
            camera_v: Vec3::new(0.0, 0.0, 0.0),
            
            mouse_grabbed: false,
            held_keys: HashSet::new(),
            last_frame: Instant::now(),
            start_time: Instant::now(),

            clear_color: wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.08,
                a: 1.0,
            }
        }
    }

    fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Sampler for shader depth
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        (texture, view, sampler)
    }

    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    // Resize when needed
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return; // Minimized
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        (self.depth_texture, self.depth_view, self.sampler) = Self::create_depth_texture(&self.device, &self.config);
    }

    // Keyboard inputs
    pub fn process_input(&mut self, key: &KeyCode, pressed: bool) {
    if pressed {
            self.held_keys.insert(*key);
        } else {
            self.held_keys.remove(key);
        }
    }

    // Mouse inputs
    pub fn process_mouse(&mut self, dx: f32, dy: f32) {
        if !self.mouse_grabbed {return}

        let sensitivity = 0.002;
        self.camera_yaw   += dx * sensitivity;
        self.camera_pitch -= dy * sensitivity;
    }
    
    pub fn set_mouse_grab(&mut self, window: &Window, grabbed: bool) {
        self.mouse_grabbed = grabbed;

        let grab_mode = if grabbed {
            winit::window::CursorGrabMode::Locked
        } else {
            winit::window::CursorGrabMode::None
        };

        if let Err(e) = window.set_cursor_grab(grab_mode) {
            log::warn!("Failed to grab cursor: {e}");
        }
        window.set_cursor_visible(!grabbed);
    }

    // Chunk updater
    pub fn update_chunks(&mut self) {
        let cam_chunk = glam::IVec3::new(
            (self.camera_pos.x / CHUNK_SIZE as f32).floor() as i32,
            (self.camera_pos.y / CHUNK_SIZE as f32).floor() as i32,
            (self.camera_pos.z / CHUNK_SIZE as f32).floor() as i32,
        );

        let radius = RENDER_DISTANCE;

        for x in -radius..=radius {
            for z in -radius..=radius {
                for y in -radius..=radius {
                    let pos = cam_chunk + glam::IVec3::new(x, y, z);
                    
                    if pos.y < Y_DIST.x || pos.y > Y_DIST.y {
                        continue;
                    }
                    
                    if self.chunks.contains_key(&pos) || self.pending_chunks.contains(&pos) {
                        continue;
                    }

                    self.pending_chunks.insert(pos);
                    let _ = self.tx_request.send(ChunkRequest::Load(pos));
                }
            }
        }

        let mut world_mesh_dirty = false;
        while let Ok(response) = self.rx_response.try_recv() {
            match response {
                ChunkResponse::Loaded(pos, chunk) => {
                    self.pending_chunks.remove(&pos);

                    // Rebuild on main thread
                    self.chunks.insert(pos, chunk);
                    world_mesh_dirty = true;
                }
            }
        }

        // Hysteresis padding
        let unload_padding = 2;
        let max_dist = RENDER_DISTANCE + unload_padding;

        let mut chunks_to_remove = Vec::new();

        // Identify chunks outside view distance
        for &pos in self.chunks.keys() {
            let delta = pos - cam_chunk;
            
            if delta.x.abs() > max_dist || delta.z.abs() > max_dist || delta.y.abs() > max_dist {
                chunks_to_remove.push(pos);
            }
        }

        if !chunks_to_remove.is_empty() {
            for pos in chunks_to_remove {
                self.chunks.remove(&pos);
                self.pending_chunks.remove(&pos);

                for offset in [
                    glam::IVec3::X, glam::IVec3::NEG_X,
                    glam::IVec3::Y, glam::IVec3::NEG_Y,
                    glam::IVec3::Z, glam::IVec3::NEG_Z,
                ] {
                    if let Some(neighbor) = self.chunks.get_mut(&(pos + offset)) {
                        neighbor.dirty = true;
                    }
                }
            }
            world_mesh_dirty = true;
        }

        let dirty_positions: Vec<glam::IVec3> = self.chunks.iter()
            .filter(|(_, chunk)| chunk.dirty)
            .map(|(&pos, _)| pos)
            .collect();

        for pos in dirty_positions {
            if let Some(mut chunk) = self.chunks.remove(&pos) {
                chunk.rebuild_mesh();
                self.chunks.insert(pos, chunk);
                world_mesh_dirty = true;
            }
        }

        if world_mesh_dirty {
            self.rebuild_super_mesh();
        }
    }

    // Chunk updates
    fn rebuild_super_mesh(&mut self) {
        let mut master_vertices: Vec<Vertex> = Vec::new();
        let mut master_indices: Vec<u32> = Vec::new();

        for chunk in self.chunks.values() {
            let vertex_offset = master_vertices.len() as u32;
            master_vertices.extend_from_slice(&chunk.vertices);
            
            for &idx in &chunk.indices {
                master_indices.push(idx + vertex_offset);
            }
        }

        if master_vertices.is_empty() {
            self.super_num_indices = 0;
            return;
        }

        self.super_num_indices = master_indices.len() as u32;

        self.queue.write_buffer(
            &self.super_vertex_buffer,
            0,
            bytemuck::cast_slice(&master_vertices),
        );

        self.queue.write_buffer(
            &self.super_index_buffer,
            0,
            bytemuck::cast_slice(&master_indices),
        );
    }

    // Per frame updates
    pub fn update(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        let time = self.start_time.elapsed().as_secs_f32();

        let speed: f32 = 60.0;
        let friction: f32 = 0.05;

        let fps = 1.0 / dt;
        print!("\rFPS: {:.0}", fps);
        io::stdout().flush().unwrap();


        // Update chunks every 0.1 seconds
        if time % 0.1 < dt {
            self.update_chunks();
        }


        let forward = Vec3::new(
            self.camera_yaw.cos() * self.camera_pitch.cos(),
            self.camera_pitch.sin(),
            self.camera_yaw.sin() * self.camera_pitch.cos(),
        ).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        

        // Keyboard input handling
        if self.held_keys.contains(&KeyCode::KeyW) {self.camera_v += forward * speed * dt}
        if self.held_keys.contains(&KeyCode::KeyS) {self.camera_v -= forward * speed * dt}
        if self.held_keys.contains(&KeyCode::KeyA) {self.camera_v -= right * speed * dt}
        if self.held_keys.contains(&KeyCode::KeyD) {self.camera_v += right * speed * dt}
        if self.held_keys.contains(&KeyCode::Space)     {self.camera_v.y += speed * dt}
        if self.held_keys.contains(&KeyCode::ShiftLeft) {self.camera_v.y -= speed * dt}

        self.camera_pos += self.camera_v * dt;
        self.camera_v *= friction.powf(dt);

        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.01;
        self.camera_pitch = self.camera_pitch.clamp(-max_pitch, max_pitch);


        let new_forward = Vec3::new(
            self.camera_yaw.cos() * self.camera_pitch.cos(),
            self.camera_pitch.sin(),
            self.camera_yaw.sin() * self.camera_pitch.cos(),
        ).normalize();

        let view = Mat4::look_at_rh(
            self.camera_pos,
            self.camera_pos + new_forward,
            Vec3::Y,
        );
        let proj = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            self.config.width as f32 / self.config.height as f32,
            0.1,
            CHUNK_SIZE as f32 * (RENDER_DISTANCE as f32 + 1.0) * 2.0,
        );
        self.camera_uniform.view_proj = (proj * view).to_cols_array_2d();


        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }
    
    // Render a frame
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        
        // Render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            if self.super_num_indices > 0 {
                render_pass.set_vertex_buffer(0, self.super_vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.super_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.super_num_indices, 0, 0..1);
            }
        }

        // Submit to the gpu
        self.queue.submit(std::iter::once(encoder.finish()));

        // Present rendered frame
        output.present();

        Ok(())
    }
}