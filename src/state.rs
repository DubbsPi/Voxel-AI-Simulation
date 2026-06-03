use glam::{
    Vec3,
    Mat4
};
use std::{
    sync::Arc,
    time::Instant
};
use wgpu::util::DeviceExt;
use winit::keyboard::KeyCode;
use std::collections::HashSet;
use winit::{dpi::PhysicalSize, window::Window};


// Vertex uniform
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
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
                    // Offset sits after position and color arrays
                    offset: (std::mem::size_of::<[f32; 3]>() * 2) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // location 2 = color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
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
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
pub enum BlockType {
    Air,
    Solid,
    // add more as needed
}

pub struct Chunk {
    pub blocks: Box<[[[BlockType; 16]; 16]; 16]>,

    // In chunk coords
    pub position: glam::IVec3,

    // GPU buffers
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    num_indices: u32,

    pub dirty: bool,
}

impl Chunk {
    pub fn new(position: glam::IVec3) -> Self {
        Self {
            blocks: Box::new([[[BlockType::Air; 16]; 16]; 16]),
            position,
            vertex_buffer: None,
            index_buffer: None,
            num_indices: 0,
            dirty: true,
        }
    }

    pub fn set_block(&mut self, x: usize, y: usize, z: usize, block: BlockType) {
        self.blocks[x][y][z] = block;
        self.dirty = true;
    }

    fn get_block_local(&self, x: i32, y: i32, z: i32) -> BlockType {
        if x < 0 || y < 0 || z < 0 || x >= 16 || y >= 16 || z >= 16 {
            return BlockType::Air;
        }
        self.blocks[x as usize][y as usize][z as usize]
    }

    pub fn rebuild_mesh(&mut self, device: &wgpu::Device) {
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u16> = Vec::new();

        let chunk_world = glam::Vec3::new(
            (self.position.x * 16) as f32,
            (self.position.y * 16) as f32,
            (self.position.z * 16) as f32,
        );

        for x in 0..16i32 {
            for y in 0..16i32 {
                for z in 0..16i32 {
                    if self.blocks[x as usize][y as usize][z as usize] == BlockType::Air {
                        continue;
                    }

                    let world_pos = chunk_world + glam::Vec3::new(x as f32, y as f32, z as f32);

                    if self.get_block_local(x + 1, y, z) == BlockType::Air {
                        add_face(&mut vertices, &mut indices, world_pos, Face::Right);
                    }
                    if self.get_block_local(x - 1, y, z) == BlockType::Air {
                        add_face(&mut vertices, &mut indices, world_pos, Face::Left);
                    }
                    if self.get_block_local(x, y + 1, z) == BlockType::Air {
                        add_face(&mut vertices, &mut indices, world_pos, Face::Top);
                    }
                    if self.get_block_local(x, y - 1, z) == BlockType::Air {
                        add_face(&mut vertices, &mut indices, world_pos, Face::Bottom);
                    }
                    if self.get_block_local(x, y, z + 1) == BlockType::Air {
                        add_face(&mut vertices, &mut indices, world_pos, Face::Front);
                    }
                    if self.get_block_local(x, y, z - 1) == BlockType::Air {
                        add_face(&mut vertices, &mut indices, world_pos, Face::Back);
                    }
                }
            }
        }

        if vertices.is_empty() {
            // All air
            self.vertex_buffer = None;
            self.index_buffer = None;
            self.num_indices = 0;
            self.dirty = false;
            return;
        }

        self.num_indices = indices.len() as u32;

        self.vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("chunk {:?} vertex buffer", self.position)),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX
        }));
        self.index_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("chunk {:?} index buffer", self.position)),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX
        }));

        self.dirty = false;
    }
}

#[derive(Clone, Copy)]
enum Face { Top, Bottom, Left, Right, Front, Back }

fn add_face(vertices: &mut Vec<Vertex>, indices: &mut Vec<u16>, pos: glam::Vec3, face: Face) {
    let base = vertices.len() as u16;

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
    
    chunks: Vec<Chunk>,

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

    // Background color
    clear_color: wgpu::Color,
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
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
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
            present_mode: wgpu::PresentMode::AutoVsync,
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

        
        // Buffers
        let mut chunks: Vec<Chunk> = Vec::new();

        let mut chunk = Chunk::new(glam::IVec3::new(0, -1, -1));
        for x in 0..16 {
            for z in 0..16 {
                let xf = x as f32;
                let zf = z as f32;
                chunk.set_block(x, ((xf + zf * 0.5).sin() * 3.0 + 10.0) as usize, z, BlockType::Solid);
            }
        }
        chunk.rebuild_mesh(&device);
        chunks.push(chunk);

        let (depth_texture, depth_view, sampler) = Self::create_depth_texture(&device, &config);
                
        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,

            chunks,

            depth_texture,
            depth_view,
            sampler,

            camera_buffer,
            camera_uniform,
            camera_bind_group,

            camera_pos: Vec3::new(0.0, 0.0, 2.0),
            camera_yaw: -std::f32::consts::FRAC_PI_2,
            camera_pitch: 0.0,
            camera_v: Vec3::new(0.0, 0.0, 0.0),
            
            mouse_grabbed: false,
            held_keys: HashSet::new(),
            last_frame: Instant::now(),

            clear_color: wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.08,
                a: 1.0,
            },
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

    fn create_vertex_buffer(device: &wgpu::Device, vertices: &[Vertex]) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn create_index_buffer(device: &wgpu::Device, indices: &[u16]) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        })
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

    // Per frame updates
    pub fn update(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;


        let speed: f32 = 15.0;
        let friction: f32 = 0.05;


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
            100.0,
        );
        self.camera_uniform.view_proj = (proj * view).to_cols_array_2d();


        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Chunk rebuilds
        for chunk in &mut self.chunks {
            if chunk.dirty {
                chunk.rebuild_mesh(&self.device);
            }
        }
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

            for chunk in &self.chunks {
                if let (Some(vb), Some(ib)) = (&chunk.vertex_buffer, &chunk.index_buffer) {
                    render_pass.set_vertex_buffer(0, vb.slice(..));
                    render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..chunk.num_indices, 0, 0..1);
                }
            }
        }

        // Submit to the gpu
        self.queue.submit(std::iter::once(encoder.finish()));

        // Present rendered frame
        output.present();

        Ok(())
    }
}