use fastnoise_lite::*;


#[derive(Clone, Copy, PartialEq)]
pub enum BlockType {
    Air,
    Solid
}


// World generation setup
pub struct WorldGen {
    base_noise: FastNoiseLite,
    detail_noise: FastNoiseLite,
    cave_noise: FastNoiseLite,

    pub sea_level: i32
}

impl WorldGen {
    pub fn new(seed: i32) -> Self {
        // Base terrain
        let mut base_noise = FastNoiseLite::with_seed(seed);
        base_noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        base_noise.set_fractal_type(Some(FractalType::FBm));
        base_noise.set_fractal_octaves(Some(5));
        base_noise.set_fractal_lacunarity(Some(2.0));
        base_noise.set_fractal_gain(Some(0.5));
        base_noise.set_frequency(Some(0.003));
        base_noise.set_domain_warp_type(Some(DomainWarpType::OpenSimplex2));
        base_noise.set_domain_warp_amp(Some(60.0));

        // Detail layer
        let mut detail_noise = FastNoiseLite::with_seed(seed + 2);
        detail_noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        detail_noise.set_frequency(Some(0.02));

        // Cave layer
        let mut cave_noise = FastNoiseLite::with_seed(seed + 3);
        cave_noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        cave_noise.set_frequency(Some(0.04));

        Self {
            base_noise,
            detail_noise,
            cave_noise,
            sea_level: 32
        }
    }

    pub fn surface_height(&self, world_x: i32, world_z: i32) -> i32 {
        let x = world_x as f32;
        let z = world_z as f32;

        // Remap base to sea level
        let base = self.base_noise.get_noise_2d(x, z);
        let height = self.sea_level as f32 + base * 40.0;

        let detail = self.detail_noise.get_noise_2d(x, z) * 5.0;

        (height + detail) as i32
    }

    fn is_cave(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        // 3D noise for caves
        let n = self.cave_noise.get_noise_3d(world_x as f32, world_y as f32, world_z as f32);
        n.abs() < 0.1 // Cave threshold
    }

    pub fn generate_chunk(&self, chunk: &mut crate::state::Chunk) {
        let cx = chunk.position.x * crate::state::CHUNK_SIZE;
        let cy = chunk.position.y * crate::state::CHUNK_SIZE;
        let cz = chunk.position.z * crate::state::CHUNK_SIZE;

        for x in 0..crate::state::PADDED_SIZE {
            for z in 0..crate::state::PADDED_SIZE {
                let world_x = cx + x as i32 - 1;
                let world_z = cz + z as i32 - 1;
                let surface = self.surface_height(world_x, world_z);

                for y in 0..crate::state::PADDED_SIZE {
                    let world_y = cy + y as i32 - 1;

                    let block = if world_y > surface {
                        BlockType::Air
                    } else if self.is_cave(world_x, world_y, world_z) {
                        BlockType::Air 
                    } else {
                        BlockType::Solid
                    };

                    chunk.blocks[x as usize][y as usize][z as usize] = block;
                }
            }
        }
    }
}