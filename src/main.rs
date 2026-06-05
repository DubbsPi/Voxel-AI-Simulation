use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
    keyboard::{KeyCode}
};

mod state;
use state::State;

pub mod worldgen;

use rand::Rng;


// Winit's current state
struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            state: None,
        }
    }
}

// Keeps winit's stuff
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Creation of the window
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Voxel AI Simulation")
                        .with_inner_size(winit::dpi::LogicalSize::new(1440, 810)),
                )
                .expect("failed to create window"),
        );

        let state = pollster::block_on(State::new(Arc::clone(&window)));

        self.state = Some(state);
        self.window = Some(window);
    }

    // For when a system suspends the app, this will stop it from breaking
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.state = None;
        self.window = None;
    }

    // This handles everything going on with the window (inputs, errors, resizes, etc)
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        let (Some(state), Some(window)) = (self.state.as_mut(), self.window.as_ref()) else {
            return;
        };

        match event {
            // Close the window
            WindowEvent::CloseRequested => {
                log::info!("Window close requested — bye!");
                event_loop.exit();
            }

            // Keyboard inputs
            WindowEvent::KeyboardInput {event, ..} => {
                if let winit::keyboard::PhysicalKey::Code(keycode) = event.physical_key {
                    if keycode == KeyCode::Escape && event.state.is_pressed() {
                        if let (Some(state), Some(window)) = (self.state.as_mut(), self.window.as_ref()) {
                            state.set_mouse_grab(window, false);
                        }
                    }

                    if let Some(state) = self.state.as_mut() {
                        state.process_input(&keycode, event.state.is_pressed());
                    }
                }
            }

            // Window resize
            WindowEvent::Resized(new_size) => {
                state.resize(new_size);
                window.request_redraw();
            }

            // Mouse grabbing
            WindowEvent::MouseInput {state: winit::event::ElementState::Pressed, ..} => {
                if let (Some(state), Some(window)) = (self.state.as_mut(), self.window.as_ref()) {
                    state.set_mouse_grab(window, true);
                }
            }


            // Per frame drawing
            WindowEvent::RedrawRequested => {
                state.update();

                match state.render() {
                    Ok(_) => {}

                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        log::warn!("Surface lost/outdated — reconfiguring");
                        state.resize(state.size());
                    }

                    // Out of memory error
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        let mut rng = rand::rng();
                        let number  = rng.random_range(1..=5);
                        
                        match number {
                            1 => log::error!("Out of GPU memory"),
                            2 => log::error!("VRAM is gone"),
                            3 => log::error!("VRAM crisis, huh?"),
                            4 => log::error!("Get more GPU memory"),
                            5 => log::error!("How did you run out of VRAM? This isn't even that much of a program!"),
                            _ => log::error!("I don't think your issue is VRAM...")
                        }
                        event_loop.exit();
                    }

                    // Other errors
                    Err(e) => log::error!("Render error: {:?}", e),
                }

                window.request_redraw();
            }

            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: winit::event::DeviceId, event: winit::event::DeviceEvent) {
        if let winit::event::DeviceEvent::MouseMotion {delta} = event {
            if let Some(state) = self.state.as_mut() {
                state.process_mouse(delta.0 as f32, delta.1 as f32);
            }
        }
    }
}


// Main
fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .filter_module("Voxel AI Simulation", log::LevelFilter::Info)
        .init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();

    event_loop
        .run_app(&mut app)
        .expect("Event loop exited with error");
}