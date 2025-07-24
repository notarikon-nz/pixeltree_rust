// Cargo.toml dependencies needed:
// [dependencies]
// bevy = "0.16.1"
// rand = { version = "0.8", features = ["small_rng"] }
// serde = { version = "1.0", features = ["derive"] }
// serde_json = "1.0"
// rayon = "1.7"

use bevy::prelude::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CORE DATA STRUCTURES
// ============================================================================

#[derive(Component, Clone, Reflect)]
pub struct PixelTree {
    pub trunk: TrunkData,
    pub branches: Vec<Branch>,
    pub leaves: LeafCluster,
    pub wind_params: WindParams,
    pub lod_level: u8,
    pub template: TreeTemplate,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub enum TreeTemplate {
    Default,
    Pine,
    Oak,
    Willow,
    Minimal,
}

#[derive(Clone, Reflect)]
pub struct Branch {
    pub start: Vec2,
    pub end: Vec2,
    pub thickness: f32,
    pub generation: u8,
    pub parent: Option<usize>,
}

#[derive(Clone, Reflect)]
pub struct TrunkData {
    pub base_pos: Vec2,
    pub height: f32,
    pub base_width: f32,
    pub segments: Vec<TrunkSegment>,
}

#[derive(Clone, Reflect)]
pub struct TrunkSegment {
    pub start: Vec2,
    pub end: Vec2,
    pub width: f32,
}

// Optimized leaf structure with better cache locality
#[derive(Component, Clone, Reflect)]
pub struct LeafCluster {
    pub leaves: Vec<Leaf>,
}

#[derive(Clone, Copy, Reflect)]
pub struct Leaf {
    pub pos: Vec2,
    pub rot: f32,
    pub scale: f32,
    pub leaf_type: u8,
    pub color: [u8; 3], // RGB bytes instead of Color
}

#[derive(Component, Clone, Reflect)]
pub struct WindParams {
    pub strength: f32,
    pub frequency: f32,
    pub turbulence: f32,
    pub direction: Vec2,
}

// Instance data for GPU instancing support
#[derive(Component)]
pub struct TreeInstanceData {
    pub transform_matrix: Mat4,
    pub wind_phase: f32,
    pub color_variation: Vec3,
}

// ============================================================================
// GENERATION PARAMETERS
// ============================================================================

#[derive(Clone)]
pub struct GenerationParams {
    pub seed: u64,
    pub height_range: (f32, f32),
    pub trunk_width_range: (f32, f32),
    pub branch_angle_variance: f32,
    pub branch_length_decay: f32,
    pub max_generations: u8,
    pub branching_probability: f32,
    pub leaf_density: f32,
    pub lod_level: u8,
    pub wind_params: WindParams,
    pub template: TreeTemplate,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            seed: 12345,
            height_range: (60.0, 120.0),
            trunk_width_range: (8.0, 16.0),
            branch_angle_variance: 45.0,
            branch_length_decay: 0.7,
            max_generations: 4,
            branching_probability: 0.8,
            leaf_density: 1.0,
            lod_level: 0,
            wind_params: WindParams {
                strength: 2.0,
                frequency: 1.5,
                turbulence: 0.3,
                direction: Vec2::new(1.0, 0.0),
            },
            template: TreeTemplate::Default,
        }
    }
}

// ============================================================================
// TREE GENERATOR
// ============================================================================

pub struct TreeGenerator {
    rng: SmallRng,
}

impl TreeGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    pub fn generate(&mut self, params: &GenerationParams) -> PixelTree {
        let trunk = self.generate_trunk(params);
        
        // Skip branch/leaf generation for minimal LOD
        if params.lod_level >= 3 {
            return PixelTree {
                trunk,
                branches: Vec::new(),
                leaves: LeafCluster { leaves: Vec::new() },
                wind_params: params.wind_params.clone(),
                lod_level: params.lod_level,
                template: TreeTemplate::Minimal,
            };
        }
        
        let branches = self.generate_branches(&trunk, params);
        let leaves = if params.lod_level >= 2 {
            LeafCluster { leaves: Vec::new() }
        } else {
            self.generate_leaves(&branches, params)
        };

        PixelTree {
            trunk,
            branches,
            leaves,
            wind_params: params.wind_params.clone(),
            lod_level: params.lod_level,
            template: params.template,
        }
    }

    fn generate_trunk(&mut self, params: &GenerationParams) -> TrunkData {
        let height = self.rng.gen_range(params.height_range.0..=params.height_range.1);
        let base_width = self.rng.gen_range(params.trunk_width_range.0..=params.trunk_width_range.1);
        
        let segment_count = (height / 20.0).max(3.0) as usize;
        let mut segments = Vec::with_capacity(segment_count);
        
        for i in 0..segment_count {
            let t = i as f32 / (segment_count - 1) as f32;
            let y_start = t * height;
            let y_end = if i == segment_count - 1 { height } else { (i + 1) as f32 / (segment_count - 1) as f32 * height };
            
            let width_start = base_width * (1.0 - t * 0.7);
            
            segments.push(TrunkSegment {
                start: Vec2::new(0.0, y_start),
                end: Vec2::new(0.0, y_end),
                width: width_start,
            });
        }

        TrunkData {
            base_pos: Vec2::ZERO,
            height,
            base_width,
            segments,
        }
    }

    // Stack-based iteration instead of recursion for better performance
    fn generate_branches(&mut self, trunk: &TrunkData, params: &GenerationParams) -> Vec<Branch> {
        // Pre-calculate capacity to avoid reallocations
        let estimated_branches = (params.max_generations as usize).pow(2) * 4;
        let mut branches = Vec::with_capacity(estimated_branches.min(64));
        let mut stack = Vec::with_capacity(32);
        
        // Generate main branches from trunk
        let main_branch_count = self.rng.gen_range(3..=6);
        for i in 0..main_branch_count {
            let height_factor = 0.6 + (i as f32 / main_branch_count as f32) * 0.4;
            let start_pos = Vec2::new(0.0, trunk.height * height_factor);
            
            let angle = self.rng.gen_range(-params.branch_angle_variance..params.branch_angle_variance).to_radians();
            let length = trunk.height * 0.4 * self.rng.gen_range(0.7..1.2);
            
            let direction = Vec2::new(angle.sin(), angle.cos().abs());
            let end_pos = start_pos + direction * length;
            
            let branch_idx = branches.len();
            branches.push(Branch {
                start: start_pos,
                end: end_pos,
                thickness: trunk.base_width * 0.4,
                generation: 1,
                parent: None,
            });
            
            // Add to stack for sub-branch generation
            if params.max_generations > 1 && self.rng.gen_range(0.0..1.0) < params.branching_probability {
                stack.push((branch_idx, 2));
            }
        }
        
        // Generate sub-branches using stack
        while let Some((parent_idx, generation)) = stack.pop() {
            if generation > params.max_generations { continue; }
            
            // Clone parent data to avoid borrow issues
            let parent_start = branches[parent_idx].start;
            let parent_end = branches[parent_idx].end;
            let parent_thickness = branches[parent_idx].thickness;
            
            let sub_branch_count = self.rng.gen_range(1..=3);
            
            for _ in 0..sub_branch_count {
                let t = self.rng.gen_range(0.3..0.9);
                let start_pos = parent_start.lerp(parent_end, t);
                
                let parent_dir = (parent_end - parent_start).normalize();
                let angle_offset = self.rng.gen_range(-60.0..60.0_f32).to_radians();
                let new_dir = Vec2::new(
                    parent_dir.x * angle_offset.cos() - parent_dir.y * angle_offset.sin(),
                    parent_dir.x * angle_offset.sin() + parent_dir.y * angle_offset.cos(),
                );
                
                let length = parent_start.distance(parent_end) * params.branch_length_decay;
                let end_pos = start_pos + new_dir * length;
                
                let branch_idx = branches.len();
                branches.push(Branch {
                    start: start_pos,
                    end: end_pos,
                    thickness: parent_thickness * 0.7,
                    generation,
                    parent: Some(parent_idx),
                });
                
                // Add children to stack
                if generation < params.max_generations && self.rng.gen_range(0.0..1.0) < params.branching_probability {
                    stack.push((branch_idx, generation + 1));
                }
            }
        }
        
        branches
    }

    fn generate_leaves(&mut self, branches: &[Branch], params: &GenerationParams) -> LeafCluster {
        let mut leaves = Vec::new();
        
        // Only add leaves to terminal branches (highest generation)
        let max_gen = branches.iter().map(|b| b.generation).max().unwrap_or(0);
        
                    for branch in branches.iter().filter(|b| b.generation >= max_gen.saturating_sub(1)) {
            let leaf_count = (branch.start.distance(branch.end) * params.leaf_density * 0.1) as usize;
            
            for _ in 0..leaf_count {
                let t = self.rng.gen_range(0.2..1.0);
                let pos = branch.start.lerp(branch.end, t);
                
                // Add some random offset
                let offset = Vec2::new(
                    self.rng.gen_range(-8.0..8.0),
                    self.rng.gen_range(-8.0..8.0),
                );
                
                let green_variation = self.rng.gen_range(0.8..1.0);
                leaves.push(Leaf {
                    pos: pos + offset,
                    rot: self.rng.gen_range(0.0..360.0),
                    scale: self.rng.gen_range(0.8..1.2),
                    leaf_type: self.rng.gen_range(0..4),
                    color: [
                        (0.2 * 255.0) as u8,
                        (green_variation * 255.0) as u8,
                        (0.1 * 255.0) as u8,
                    ],
                });
            }
        }
        
        LeafCluster { leaves }
    }
}

// Deterministic tree generation without RNG state
pub fn generate_tree_deterministic(pos: Vec2, params: &GenerationParams) -> PixelTree {
    let seed = hash_position(pos) ^ params.seed;
    let mut generator = TreeGenerator::new(seed);
    generator.generate(params)
}

#[inline]
fn hash_position(pos: Vec2) -> u64 {
    let x_bits = pos.x.to_bits() as u64;
    let y_bits = pos.y.to_bits() as u64;
    x_bits.wrapping_mul(0x45d9f3b).wrapping_add(y_bits.wrapping_mul(0x119de1f3))
}

// ============================================================================
// BATCH GENERATION
// ============================================================================

use rayon::prelude::*;

pub struct BatchTreeGenerator {
    base_params: GenerationParams,
}

impl BatchTreeGenerator {
    pub fn new(base_params: GenerationParams) -> Self {
        Self { base_params }
    }

    pub fn generate_forest(&self, positions: &[Vec2], seed_offsets: &[u64]) -> Vec<(Vec2, PixelTree)> {
        positions
            .par_iter()
            .zip(seed_offsets.par_iter())
            .map(|(pos, seed_offset)| {
                let mut params = self.base_params.clone();
                params.seed = params.seed.wrapping_add(*seed_offset);
                
                let mut generator = TreeGenerator::new(params.seed);
                let tree = generator.generate(&params);
                
                (*pos, tree)
            })
            .collect()
    }
}

// ============================================================================
// LOD SYSTEM
// ============================================================================

#[derive(Clone, Copy, PartialEq)]
pub enum LodLevel {
    High = 0,
    Medium = 1,
    Low = 2,
    Minimal = 3,
}

#[derive(Resource)]
pub struct LodConfig {
    pub max_branches: [usize; 4],
    pub leaf_density: [f32; 4],
    pub detail_distance: [f32; 4],
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            max_branches: [64, 32, 8, 0],
            leaf_density: [1.0, 0.6, 0.3, 0.0],
            detail_distance: [50.0, 100.0, 200.0, 400.0],
        }
    }
}

pub fn calculate_lod_level(distance: f32, config: &LodConfig) -> LodLevel {
    for (i, &max_dist) in config.detail_distance.iter().enumerate() {
        if distance <= max_dist {
            return match i {
                0 => LodLevel::High,
                1 => LodLevel::Medium,
                2 => LodLevel::Low,
                _ => LodLevel::Minimal,
            };
        }
    }
    LodLevel::Minimal
}

// ============================================================================
// WIND ANIMATION WITH PRECOMPUTED TABLE
// ============================================================================

#[derive(Resource)]
pub struct WindTable {
    samples: [f32; 256],
}

impl Default for WindTable {
    fn default() -> Self {
        Self::new()
    }
}

impl WindTable {
    pub fn new() -> Self {
        let mut samples = [0.0; 256];
        for (i, s) in samples.iter_mut().enumerate() {
            *s = (i as f32 * std::f32::consts::TAU / 256.0).sin();
        }
        Self { samples }
    }
    
    #[inline(always)]
    pub fn sample(&self, phase: f32) -> f32 {
        let idx = ((phase % std::f32::consts::TAU) * 256.0 / std::f32::consts::TAU) as usize;
        self.samples[idx.min(255)]
    }
}

// Simplified wind calculation
#[inline(always)]
fn calculate_wind_displacement(time: f32, wind: &WindParams, pos: Vec2, wind_table: &WindTable) -> Vec2 {
    let phase = time * wind.frequency + pos.dot(Vec2::new(0.1, 0.15));
    wind.direction * wind.strength * wind_table.sample(phase) * (1.0 + wind.turbulence)
}

// ============================================================================
// SPATIAL INDEXING FOR CULLING
// ============================================================================

#[derive(Resource)]
pub struct TreeSpatialIndex {
    cell_size: f32,
    cells: HashMap<(i32, i32), Vec<Entity>>,
}

impl Default for TreeSpatialIndex {
    fn default() -> Self {
        Self::new(100.0)
    }
}

impl TreeSpatialIndex {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            cells: HashMap::new(),
        }
    }
    
    pub fn insert(&mut self, entity: Entity, pos: Vec2) {
        let cell = self.world_to_cell(pos);
        self.cells.entry(cell).or_insert_with(Vec::new).push(entity);
    }
    
    pub fn get_visible_trees(&self, view_bounds: Rect) -> Vec<Entity> {
        let min_cell = self.world_to_cell(view_bounds.min);
        let max_cell = self.world_to_cell(view_bounds.max);
        
        let mut visible = Vec::with_capacity(128);
        for x in min_cell.0..=max_cell.0 {
            for y in min_cell.1..=max_cell.1 {
                if let Some(entities) = self.cells.get(&(x, y)) {
                    visible.extend(entities);
                }
            }
        }
        visible
    }
    
    #[inline]
    fn world_to_cell(&self, pos: Vec2) -> (i32, i32) {
        (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
        )
    }
}

// ============================================================================
// TEMPLATE CACHE FOR INSTANCING
// ============================================================================

#[derive(Resource)]
pub struct TreeTemplateCache {
    templates: HashMap<TreeTemplate, TreeMeshData>,
}

impl Default for TreeTemplateCache {
    fn default() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }
}

pub struct TreeMeshData {
    pub vertex_data: Vec<TreeVertex>,
    pub index_data: Vec<u16>,
}

#[derive(Clone, Copy)]
pub struct TreeVertex {
    pub position: Vec2,
    pub uv: Vec2,
    pub color: [u8; 4],
}

// ============================================================================
// BEVY INTEGRATION
// ============================================================================

pub struct PixelTreePlugin;

impl Plugin for PixelTreePlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Update, update_trees)
            .init_resource::<LodConfig>()
            .init_resource::<WindTable>()
            .init_resource::<TreeSpatialIndex>()
            .init_resource::<TreeTemplateCache>()
            .register_type::<PixelTree>()
            .register_type::<WindParams>()
            .register_type::<LeafCluster>();
    }
}

// Combined update system for better performance
pub fn update_trees(
    time: Res<Time>,
    wind_table: Res<WindTable>,
    camera_query: Query<&Transform, (With<Camera>, Without<PixelTree>)>,
    mut tree_query: Query<(&mut Transform, &mut PixelTree)>,
    lod_config: Res<LodConfig>,
) {
    if let Ok(camera_transform) = camera_query.single() {
        let camera_pos = camera_transform.translation.truncate();
        let elapsed = time.elapsed_secs();
        
        for (mut transform, mut tree) in tree_query.iter_mut() {
            let tree_pos = transform.translation.truncate();
            
            // Update LOD
            let distance = camera_pos.distance(tree_pos);
            let new_lod = calculate_lod_level(distance, &lod_config) as u8;
            
            if tree.lod_level != new_lod {
                tree.lod_level = new_lod;
                // Mark for regeneration or LOD swap
            }
            
            // Apply wind animation
            let wind_offset = calculate_wind_displacement(
                elapsed,
                &tree.wind_params,
                tree_pos,
                &wind_table,
            );
            
            transform.rotation = Quat::from_rotation_z(wind_offset.x * 0.01);
        }
    }
}

// ============================================================================
// UTILITY FUNCTIONS FOR SPAWNING
// ============================================================================

pub fn spawn_tree(
    commands: &mut Commands,
    position: Vec3,
    params: GenerationParams,
    spatial_index: &mut TreeSpatialIndex,
) -> Entity {
    let mut generator = TreeGenerator::new(params.seed);
    let tree = generator.generate(&params);
    
    let entity = commands.spawn((
        Transform::from_translation(position),
        GlobalTransform::default(),
        tree,
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
    )).id();
    
    spatial_index.insert(entity, position.truncate());
    entity
}

pub fn spawn_forest(
    commands: &mut Commands,
    positions: Vec<Vec2>,
    base_params: GenerationParams,
    spatial_index: &mut TreeSpatialIndex,
) {
    let batch_generator = BatchTreeGenerator::new(base_params);
    let seed_offsets: Vec<u64> = (0..positions.len()).map(|i| i as u64 * 1337).collect();
    
    let trees = batch_generator.generate_forest(&positions, &seed_offsets);
    
    for (pos, tree) in trees {
        let entity = commands.spawn((
            Transform::from_translation(pos.extend(0.0)),
            GlobalTransform::default(),
            tree,
            Visibility::default(),
            InheritedVisibility::default(),
            ViewVisibility::default(),
        )).id();
        
        spatial_index.insert(entity, pos);
    }
}

// ============================================================================
// SERIALIZATION
// ============================================================================

#[derive(Serialize, Deserialize)]
pub struct PackedTree {
    pub trunk_height: u16,
    pub trunk_width: u8,
    pub branch_data: Vec<u32>,
    pub leaf_clusters: Vec<PackedLeafCluster>,
}

#[derive(Serialize, Deserialize)]
pub struct PackedLeafCluster {
    pub count: u16,
    pub center: [i16; 2],
    pub data: Vec<u32>, // Packed leaf data
}

impl PackedTree {
    pub fn pack(tree: &PixelTree) -> Self {
        let mut branch_data = Vec::with_capacity(tree.branches.len());
        for branch in &tree.branches {
            // Pack branch data into u32: 
            // [8 bits start_x][8 bits start_y][8 bits end_x][8 bits end_y]
            let packed = ((branch.start.x as i8 as u32) << 24)
                | ((branch.start.y as i8 as u32) << 16)
                | ((branch.end.x as i8 as u32) << 8)
                | (branch.end.y as i8 as u32);
            branch_data.push(packed);
        }
        
        Self {
            trunk_height: tree.trunk.height as u16,
            trunk_width: tree.trunk.base_width as u8,
            branch_data,
            leaf_clusters: vec![], // TODO: Implement leaf packing
        }
    }
}

// ============================================================================
// EXAMPLE USAGE
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_generation() {
        let params = GenerationParams::default();
        let mut generator = TreeGenerator::new(12345);
        let tree = generator.generate(&params);
        
        assert!(!tree.branches.is_empty());
        assert!(!tree.leaves.leaves.is_empty());
        assert_eq!(tree.lod_level, 0);
    }

    #[test]
    fn test_batch_generation() {
        let positions = vec![Vec2::ZERO, Vec2::new(100.0, 0.0), Vec2::new(0.0, 100.0)];
        let seed_offsets = vec![0, 1000, 2000];
        let batch_gen = BatchTreeGenerator::new(GenerationParams::default());
        
        let forest = batch_gen.generate_forest(&positions, &seed_offsets);
        assert_eq!(forest.len(), 3);
    }
    
    #[test]
    fn test_deterministic_generation() {
        let params = GenerationParams::default();
        let tree1 = generate_tree_deterministic(Vec2::new(100.0, 50.0), &params);
        let tree2 = generate_tree_deterministic(Vec2::new(100.0, 50.0), &params);
        
        assert_eq!(tree1.branches.len(), tree2.branches.len());
        assert_eq!(tree1.trunk.height, tree2.trunk.height);
    }
}

// Example system to demonstrate usage:
pub fn setup_demo_forest(
    mut commands: Commands,
    mut spatial_index: ResMut<TreeSpatialIndex>,
) {
    let positions = vec![
        Vec2::new(-100.0, 0.0),
        Vec2::new(0.0, 0.0),
        Vec2::new(100.0, 0.0),
        Vec2::new(-50.0, 80.0),
        Vec2::new(50.0, 80.0),
    ];
    
    let mut params = GenerationParams::default();
    params.wind_params.strength = 1.5;
    params.wind_params.frequency = 2.0;
    
    spawn_forest(&mut commands, positions, params, &mut spatial_index);
}