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
use std::path::Path;
use thiserror::Error;
use chrono::{DateTime, Utc};

// ============================================================================
// CORE DATA STRUCTURES
// ============================================================================

#[derive(Component, Clone, Debug, Reflect)]
pub struct PixelTree {
    pub trunk: TrunkData,
    pub branches: Vec<Branch>,
    pub leaves: LeafCluster,
    pub wind_params: WindParams,
    pub lod_level: u8,
    pub template: TreeTemplate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum TreeTemplate {
    Default,
    Pine,
    Oak,
    Willow,
    Minimal,
}

#[derive(Clone, PartialEq, Eq, Hash, Reflect)]
pub enum TreeSource {
    Procedural(TreeTemplate),
    Imported(String), // Path to SRT/ST file
}

#[derive(Clone, Debug, Reflect)]
pub struct Branch {
    pub start: Vec2,
    pub end: Vec2,
    pub thickness: f32,
    pub generation: u8,
    pub parent: Option<usize>,
}

#[derive(Clone, Debug, Reflect)]
pub struct TrunkData {
    pub base_pos: Vec2,
    pub height: f32,
    pub base_width: f32,
    pub segments: Vec<TrunkSegment>,
}

#[derive(Clone, Debug, Reflect)]
pub struct TrunkSegment {
    pub start: Vec2,
    pub end: Vec2,
    pub width: f32,
}

// Optimized leaf structure with better cache locality
#[derive(Component, Clone, Debug, Reflect)]
pub struct LeafCluster {
    pub leaves: Vec<Leaf>,
}

#[derive(Clone, Copy, Debug, Reflect)]
pub struct Leaf {
    pub pos: Vec2,
    pub rot: f32,
    pub scale: f32,
    pub leaf_type: u8,
    pub color: [u8; 3], // RGB bytes instead of Color
}

#[derive(Component, Clone, Debug, Default, Reflect, Serialize, Deserialize)]
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
// SPEEDTREE IMPORT STRUCTURES
// ============================================================================

#[derive(Clone, Debug)]
pub struct SpeedTreeInstance {
    pub position: Vec3,
    pub rotation: f32,
    pub scale: f32,
    pub tree_file: String,
}

#[derive(Debug)]
pub struct ForestLayout {
    pub instances: Vec<SpeedTreeInstance>,
    pub name: String,
}

#[derive(Debug)]
pub struct ImportedTreeData {
    pub trunk_segments: Vec<TrunkSegment>,
    pub branches: Vec<Branch>,
    pub lod_levels: Vec<LodData>,
    pub source_file: String,
}

#[derive(Debug)]
pub struct LodData {
    pub level: u8,
    pub max_distance: f32,
    pub branch_count: usize,
    pub leaf_density: f32,
}

#[derive(Error, Debug)]
pub enum ImportError {
    #[error("Failed to read file: {0}")]
    FileError(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

// ============================================================================
// GENERATION PARAMETERS
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
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

    pub fn generate_forest_from_stf<P: AsRef<Path>>(
        &self,
        stf_path: P,
    ) -> Result<Vec<(Vec3, PixelTree)>, ImportError> {
        let forest_layout = StfForestLoader::load_stf(stf_path)?;
        
        let trees: Vec<(Vec3, PixelTree)> = forest_layout
            .instances
            .par_iter()
            .map(|instance| {
                let filename_hash = Self::hash_string(&instance.tree_file);
                let position_hash = hash_position(instance.position.truncate());
                
                let mut params = self.base_params.clone();
                params.seed = params.seed.wrapping_add(filename_hash).wrapping_add(position_hash);
                
                // Apply STF scale to tree parameters
                params.height_range.0 *= instance.scale;
                params.height_range.1 *= instance.scale;
                params.trunk_width_range.0 *= instance.scale;
                params.trunk_width_range.1 *= instance.scale;
                
                let mut generator = TreeGenerator::new(params.seed);
                let mut tree = generator.generate(&params);
                
                // Apply rotation influence to wind direction
                let rotated_wind_dir = Vec2::new(
                    instance.rotation.cos(),
                    instance.rotation.sin(),
                );
                tree.wind_params.direction = rotated_wind_dir.normalize_or_zero();
                
                (instance.position, tree)
            })
            .collect();
        
        Ok(trees)
    }

    pub fn generate_mixed_forest(
        &self,
        procedural_positions: &[Vec2],
        imported_trees: &[ImportedTreeData],
        imported_positions: &[Vec3],
    ) -> Result<Vec<(Vec3, PixelTree)>, ImportError> {
        let mut all_trees = Vec::new();
        
        // Generate procedural trees
        let seed_offsets: Vec<u64> = (0..procedural_positions.len()).map(|i| i as u64 * 1337).collect();
        let procedural_trees = self.generate_forest(procedural_positions, &seed_offsets);
        
        for (pos, tree) in procedural_trees {
            all_trees.push((pos.extend(0.0), tree));
        }
        
        // Add imported trees
        for (imported_data, pos) in imported_trees.iter().zip(imported_positions.iter()) {
            let tree = SpeedTreeImporter::create_tree_from_imported_data(
                imported_data,
                self.base_params.wind_params.clone(),
                self.base_params.lod_level,
            );
            all_trees.push((*pos, tree));
        }
        
        Ok(all_trees)
    }

    fn hash_string(s: &str) -> u64 {
        let mut hash = 5381u64;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
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

pub fn spawn_forest_from_stf<P: AsRef<Path>>(
    commands: &mut Commands,
    stf_path: P,
    base_params: GenerationParams,
    spatial_index: &mut TreeSpatialIndex,
) -> Result<Vec<Entity>, ImportError> {
    let batch_generator = BatchTreeGenerator::new(base_params);
    let trees = batch_generator.generate_forest_from_stf(stf_path)?;
    
    let mut entities = Vec::with_capacity(trees.len());
    
    for (pos, tree) in trees {
        let entity = commands.spawn((
            Transform::from_translation(pos),
            GlobalTransform::default(),
            tree,
            Visibility::default(),
            InheritedVisibility::default(),
            ViewVisibility::default(),
        )).id();
        
        spatial_index.insert(entity, pos.truncate());
        entities.push(entity);
    }
    
    Ok(entities)
}

pub fn spawn_imported_tree<P: AsRef<Path>>(
    commands: &mut Commands,
    srt_path: P,
    position: Vec3,
    wind_params: WindParams,
    lod_level: u8,
    spatial_index: &mut TreeSpatialIndex,
) -> Result<Entity, ImportError> {
    let imported_data = SpeedTreeImporter::load_srt(srt_path)?;
    let tree = SpeedTreeImporter::create_tree_from_imported_data(&imported_data, wind_params, lod_level);
    
    let entity = commands.spawn((
        Transform::from_translation(position),
        GlobalTransform::default(),
        tree,
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
    )).id();
    
    spatial_index.insert(entity, position.truncate());
    Ok(entity)
}

// ============================================================================
// SPEEDTREE IMPORT PARSERS
// ============================================================================

// Note: nom import removed as we're using manual parsing for better control and error handling

pub struct StfForestLoader;

impl StfForestLoader {
    pub fn load_stf<P: AsRef<Path>>(path: P) -> Result<ForestLayout, ImportError> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let name = path.as_ref()
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        
        Self::parse_stf_content(&content, name)
    }

    fn parse_stf_content(content: &str, name: String) -> Result<ForestLayout, ImportError> {
        let mut instances = Vec::new();
        let mut lines = content.lines().peekable();
        
        while lines.peek().is_some() {
            if let Some(tree_instances) = Self::parse_tree_group(&mut lines)? {
                instances.extend(tree_instances);
            }
        }
        
        Ok(ForestLayout { instances, name })
    }

    fn parse_tree_group<'a, I>(lines: &mut std::iter::Peekable<I>) -> Result<Option<Vec<SpeedTreeInstance>>, ImportError>
    where
        I: Iterator<Item = &'a str>,
    {
        let tree_filename = match lines.next() {
            Some(line) if !line.trim().is_empty() => line.trim().to_string(),
            Some(_) => return Ok(None), // Empty line, skip
            None => return Ok(None), // End of file
        };

        let instance_count = match lines.next() {
            Some(line) => line.trim().parse::<usize>()
                .map_err(|_| ImportError::ParseError(format!("Invalid instance count: {}", line)))?,
            None => return Err(ImportError::ParseError("Expected instance count".to_string())),
        };

        let mut instances = Vec::with_capacity(instance_count);
        
        for _ in 0..instance_count {
            if let Some(line) = lines.next() {
                match Self::parse_instance_line(line, &tree_filename) {
                    Ok(instance) => instances.push(instance),
                    Err(e) => return Err(ImportError::ParseError(format!("Failed to parse instance: {}", e))),
                }
            } else {
                return Err(ImportError::ParseError("Unexpected end of file".to_string()));
            }
        }

        Ok(Some(instances))
    }

    fn parse_instance_line(line: &str, tree_file: &str) -> Result<SpeedTreeInstance, ImportError> {
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        
        if parts.len() != 5 {
            return Err(ImportError::ParseError(
                format!("Expected 5 values (x y z rotation scale), got {}", parts.len())
            ));
        }

        let x = parts[0].parse::<f32>()
            .map_err(|_| ImportError::ParseError(format!("Invalid x coordinate: {}", parts[0])))?;
        let y = parts[1].parse::<f32>()
            .map_err(|_| ImportError::ParseError(format!("Invalid y coordinate: {}", parts[1])))?;
        let z = parts[2].parse::<f32>()
            .map_err(|_| ImportError::ParseError(format!("Invalid z coordinate: {}", parts[2])))?;
        let rotation = parts[3].parse::<f32>()
            .map_err(|_| ImportError::ParseError(format!("Invalid rotation: {}", parts[3])))?;
        let scale = parts[4].parse::<f32>()
            .map_err(|_| ImportError::ParseError(format!("Invalid scale: {}", parts[4])))?;

        Ok(SpeedTreeInstance {
            position: Vec3::new(x, y, z),
            rotation,
            scale,
            tree_file: tree_file.to_string(),
        })
    }

    pub fn spawn_from_stf(
        commands: &mut Commands,
        forest_layout: &ForestLayout,
        base_params: GenerationParams,
        spatial_index: &mut TreeSpatialIndex,
    ) -> Vec<Entity> {
        let mut entities = Vec::with_capacity(forest_layout.instances.len());
        
        for instance in &forest_layout.instances {
            // Use the tree filename as a seed modifier to get variation
            let filename_hash = Self::hash_string(&instance.tree_file);
            let position_hash = hash_position(instance.position.truncate());
            
            let mut params = base_params.clone();
            params.seed = params.seed.wrapping_add(filename_hash).wrapping_add(position_hash);
            
            // Apply STF scale to tree parameters
            params.height_range.0 *= instance.scale;
            params.height_range.1 *= instance.scale;
            params.trunk_width_range.0 *= instance.scale;
            params.trunk_width_range.1 *= instance.scale;
            
            // Generate tree with modified parameters
            let mut generator = TreeGenerator::new(params.seed);
            let tree = generator.generate(&params);
            
            // Apply rotation from STF
            let transform = Transform::from_translation(instance.position)
                .with_rotation(Quat::from_rotation_z(instance.rotation));
            
            let entity = commands.spawn((
                transform,
                GlobalTransform::default(),
                tree,
                Visibility::default(),
                InheritedVisibility::default(),
                ViewVisibility::default(),
            )).id();
            
            spatial_index.insert(entity, instance.position.truncate());
            entities.push(entity);
        }
        
        entities
    }

    fn hash_string(s: &str) -> u64 {
        // Simple string hash for deterministic tree variation
        let mut hash = 5381u64;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }
}

// ============================================================================
// SRT/ST BINARY PARSER (Basic Structure)
// ============================================================================

pub struct SpeedTreeImporter;

impl SpeedTreeImporter {
    pub fn load_srt<P: AsRef<Path>>(path: P) -> Result<ImportedTreeData, ImportError> {
        let data = std::fs::read(path.as_ref())?;
        let source_file = path.as_ref().to_string_lossy().to_string();
        
        // Basic SRT parsing - this is a simplified version
        // Real SRT files have complex binary structures that would need reverse engineering
        Self::parse_srt_data(&data, source_file)
    }

    fn parse_srt_data(data: &[u8], source_file: String) -> Result<ImportedTreeData, ImportError> {
        // This is a placeholder implementation
        // Real SRT parsing would need detailed binary format analysis
        
        if data.len() < 16 {
            return Err(ImportError::InvalidData("File too small to be valid SRT".to_string()));
        }
        
        // For now, create a basic tree structure as fallback
        // In a real implementation, this would parse the binary SRT format
        let trunk_segments = vec![
            TrunkSegment {
                start: Vec2::new(0.0, 0.0),
                end: Vec2::new(0.0, 50.0),
                width: 8.0,
            },
            TrunkSegment {
                start: Vec2::new(0.0, 50.0),
                end: Vec2::new(0.0, 100.0),
                width: 6.0,
            },
        ];
        
        let branches = vec![
            Branch {
                start: Vec2::new(0.0, 60.0),
                end: Vec2::new(30.0, 80.0),
                thickness: 3.0,
                generation: 1,
                parent: None,
            },
        ];
        
        let lod_levels = vec![
            LodData { level: 0, max_distance: 50.0, branch_count: 32, leaf_density: 1.0 },
            LodData { level: 1, max_distance: 100.0, branch_count: 16, leaf_density: 0.6 },
            LodData { level: 2, max_distance: 200.0, branch_count: 8, leaf_density: 0.3 },
            LodData { level: 3, max_distance: 400.0, branch_count: 0, leaf_density: 0.0 },
        ];
        
        Ok(ImportedTreeData {
            trunk_segments,
            branches,
            lod_levels,
            source_file,
        })
    }

    pub fn create_tree_from_imported_data(
        data: &ImportedTreeData,
        wind_params: WindParams,
        lod_level: u8,
    ) -> PixelTree {
        let trunk = TrunkData {
            base_pos: Vec2::ZERO,
            height: data.trunk_segments.last().map(|s| s.end.y).unwrap_or(100.0),
            base_width: data.trunk_segments.first().map(|s| s.width).unwrap_or(8.0),
            segments: data.trunk_segments.clone(),
        };

        let branches = if lod_level >= 3 {
            Vec::new()
        } else {
            let lod_data = data.lod_levels.iter()
                .find(|l| l.level == lod_level)
                .unwrap_or(&data.lod_levels[0]);
            
            data.branches.iter()
                .take(lod_data.branch_count)
                .cloned()
                .collect()
        };

        let leaves = if lod_level >= 2 {
            LeafCluster { leaves: Vec::new() }
        } else {
            // Generate basic leaves for imported trees
            LeafCluster { 
                leaves: branches.iter()
                    .filter(|b| b.generation >= 2)
                    .flat_map(|b| {
                        (0..3).map(move |i| {
                            let t = (i as f32 + 1.0) / 4.0;
                            let pos = b.start.lerp(b.end, t);
                            Leaf {
                                pos,
                                rot: i as f32 * 120.0,
                                scale: 1.0,
                                leaf_type: i as u8 % 4,
                                color: [50, 200, 50],
                            }
                        })
                    })
                    .collect()
            }
        };

        PixelTree {
            trunk,
            branches,
            leaves,
            wind_params,
            lod_level,
            template: TreeTemplate::Default, // Imported trees use Default template
        }
    }
}

// ============================================================================
// CUSTOM PIXELTREE FORMAT (.pt)
// ============================================================================

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PixelTreeFile {
    pub header: PtFileHeader,
    pub procedural_params: GenerationParams,
    pub geometry_data: Option<PtGeometryData>,
    pub material_data: PtMaterialData,
    pub lod_levels: Vec<PtLodLevel>,
    pub wind_data: WindParams,
    pub metadata: PtMetadata,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtFileHeader {
    pub magic: String, // "PIXELTREE"
    pub version: String, // "0.1.0"
    pub format_version: u32, // File format version
    pub compression: bool, // Whether data is compressed
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtGeometryData {
    pub trunk_segments: Vec<PtTrunkSegment>,
    pub branches: Vec<PtBranch>,
    pub leaves: Vec<PtLeaf>,
    pub bounding_box: PtBoundingBox,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtTrunkSegment {
    pub start: [f32; 2], // Vec2 as array for better serialization
    pub end: [f32; 2],
    pub width: f32,
    pub color: [u8; 3], // RGB
    pub texture_coords: Option<[f32; 4]>, // UV mapping data
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtBranch {
    pub start: [f32; 2],
    pub end: [f32; 2],
    pub thickness: f32,
    pub generation: u8,
    pub parent: Option<usize>,
    pub color: [u8; 3],
    pub texture_coords: Option<[f32; 4]>,
    pub wind_factor: f32, // How much this branch responds to wind
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtLeaf {
    pub position: [f32; 2],
    pub rotation: f32,
    pub scale: f32,
    pub leaf_type: u8,
    pub color: [u8; 3],
    pub wind_factor: f32,
    pub season_factor: f32, // For seasonal color changes
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtBoundingBox {
    pub min: [f32; 2],
    pub max: [f32; 2],
    pub height: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtMaterialData {
    pub trunk_materials: Vec<PtMaterial>,
    pub branch_materials: Vec<PtMaterial>,
    pub leaf_materials: Vec<PtMaterial>,
    pub bark_texture: Option<String>, // Path to texture file
    pub leaf_texture: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtMaterial {
    pub name: String,
    pub base_color: [u8; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub emission: [u8; 3],
    pub texture_path: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtLodLevel {
    pub level: u8,
    pub max_distance: f32,
    pub branch_count: usize,
    pub leaf_density: f32,
    pub detail_reduction: f32, // 0.0 = full detail, 1.0 = minimal detail
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtMetadata {
    pub tree_name: String,
    pub author: String,
    pub created_date: DateTime<Utc>,
    pub modified_date: DateTime<Utc>,
    pub description: String,
    pub tags: Vec<String>,
    pub tree_type: TreeTemplate,
    pub estimated_age_years: Option<f32>, // Simulated tree age
    pub biome: Option<String>, // Forest, desert, etc.
    pub custom_properties: HashMap<String, String>,
}

// Error types for PT format
#[derive(Error, Debug)]
pub enum PtFormatError {
    #[error("Invalid file header: {0}")]
    InvalidHeader(String),
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(String),
    #[error("Compression error: {0}")]
    CompressionError(String),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("File too large: {size} bytes exceeds limit of {limit} bytes")]
    FileTooLarge { size: u64, limit: u64 },
    #[error("Invalid file path: {0}")]
    InvalidPath(String),
    #[error("Data corruption detected: {0}")]
    DataCorruption(String),
    #[error("Invalid tree data: {0}")]
    InvalidTreeData(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}

// ============================================================================
// PT FORMAT IMPLEMENTATION
// ============================================================================

// Security and validation constants
// PT Format Constants
const MAX_PT_FILE_SIZE: u64 = 100 * 1024 * 1024; // 100MB max file size
const MIN_PT_FILE_SIZE: u64 = 32; // Minimum viable file size
const MAX_TREE_NAME_LEN: usize = 256;
const MAX_AUTHOR_LEN: usize = 128;
const MAX_DESCRIPTION_LEN: usize = 1024;
const MAX_CUSTOM_PROPERTIES: usize = 50;
const MAX_PROPERTY_KEY_LEN: usize = 64;
const MAX_PROPERTY_VALUE_LEN: usize = 256;
const MAX_BRANCHES: usize = 10000;
const MAX_LEAVES: usize = 50000;
const MAX_TRUNK_SEGMENTS: usize = 1000;

// PTF Format Constants
const MAX_PTF_FILE_SIZE: u64 = 500 * 1024 * 1024; // 500MB max forest file size
const MIN_PTF_FILE_SIZE: u64 = 64; // Minimum viable PTF file size
const MAX_TREE_GROUPS: usize = 1000;
const MAX_TREE_INSTANCES: usize = 100000;
const MAX_INSTANCES_PER_GROUP: usize = 50000;
const MAX_PTF_TAGS: usize = 1000;
const MAX_PTF_CUSTOM_PROPERTIES: usize = 10000;
const MAX_TREE_FILE_PATH_LENGTH: usize = 512;

pub struct PtFileManager;

impl PtFileManager {
    pub fn save_tree<P: AsRef<Path>>(
        tree: &PixelTree,
        params: &GenerationParams,
        path: P,
        metadata: Option<PtMetadata>,
    ) -> Result<(), PtFormatError> {
        // Input validation
        Self::validate_path(&path)?;
        Self::validate_tree(tree)?;
        Self::validate_params(params)?;
        
        let geometry_data = Self::convert_tree_to_geometry(tree);
        let material_data = Self::extract_materials(tree);
        let lod_levels = Self::create_lod_levels(tree);
        
        let mut validated_metadata = metadata.unwrap_or_else(|| Self::default_metadata(&params.template));
        Self::validate_and_sanitize_metadata(&mut validated_metadata)?;
        
        let pt_file = PixelTreeFile {
            header: PtFileHeader {
                magic: "PIXELTREE".to_string(),
                version: "0.1.0".to_string(),
                format_version: 1,
                compression: true,
            },
            procedural_params: params.clone(),
            geometry_data: Some(geometry_data),
            material_data,
            lod_levels,
            wind_data: tree.wind_params.clone(),
            metadata: validated_metadata,
        };
        
        let serialized = if pt_file.header.compression {
            // Use bincode with compression for efficiency
            bincode::serialize(&pt_file).map_err(|e| {
                PtFormatError::SerializationError(e)
            })?
        } else {
            // For debugging, use JSON
            serde_json::to_vec_pretty(&pt_file).map_err(|e| {
                PtFormatError::SerializationError(bincode::Error::new(bincode::ErrorKind::Custom(e.to_string())))
            })?
        };
        
        // Check serialized size before writing
        if serialized.len() as u64 > MAX_PT_FILE_SIZE {
            return Err(PtFormatError::FileTooLarge { 
                size: serialized.len() as u64, 
                limit: MAX_PT_FILE_SIZE 
            });
        }
        
        // Atomic write - write to temporary file first, then rename
        let temp_path = path.as_ref().with_extension("pt.tmp");
        std::fs::write(&temp_path, &serialized).map_err(|e| {
            // Clean up temp file on error
            let _ = std::fs::remove_file(&temp_path);
            PtFormatError::IoError(e)
        })?;
        
        // Atomic rename
        std::fs::rename(&temp_path, &path).map_err(|e| {
            // Clean up temp file on error
            let _ = std::fs::remove_file(&temp_path);
            PtFormatError::IoError(e)
        })?;
        
        Ok(())
    }
    
    pub fn load_tree<P: AsRef<Path>>(path: P) -> Result<(PixelTree, GenerationParams), PtFormatError> {
        // Input validation
        Self::validate_path(&path)?;
        
        // Check file size
        let metadata = std::fs::metadata(path.as_ref()).map_err(PtFormatError::IoError)?;
        let file_size = metadata.len();
        
        if file_size < MIN_PT_FILE_SIZE {
            return Err(PtFormatError::DataCorruption(
                format!("File too small: {} bytes", file_size)
            ));
        }
        
        if file_size > MAX_PT_FILE_SIZE {
            return Err(PtFormatError::FileTooLarge { 
                size: file_size, 
                limit: MAX_PT_FILE_SIZE 
            });
        }
        
        let data = std::fs::read(path.as_ref()).map_err(PtFormatError::IoError)?;
        
        // Verify data integrity
        if data.len() as u64 != file_size {
            return Err(PtFormatError::DataCorruption(
                "File size mismatch during read".to_string()
            ));
        }
        
        // Try binary first, then JSON fallback
        let pt_file: PixelTreeFile = match bincode::deserialize(&data) {
            Ok(file) => file,
            Err(_) => {
                // Try JSON fallback
                serde_json::from_slice(&data).map_err(|e| {
                    PtFormatError::SerializationError(bincode::Error::new(bincode::ErrorKind::Custom(e.to_string())))
                })?
            }
        };
        
        // Validate header
        Self::validate_file_header(&pt_file.header)?;
        
        // Validate loaded data
        Self::validate_loaded_data(&pt_file)?;
        
        let tree = if let Some(geometry) = pt_file.geometry_data {
            let converted_tree = Self::convert_geometry_to_tree(geometry, &pt_file.wind_data, &pt_file.metadata);
            Self::validate_tree(&converted_tree)?;
            converted_tree
        } else {
            // Regenerate from procedural parameters
            Self::validate_params(&pt_file.procedural_params)?;
            let mut generator = TreeGenerator::new(pt_file.procedural_params.seed);
            let generated_tree = generator.generate(&pt_file.procedural_params);
            Self::validate_tree(&generated_tree)?;
            generated_tree
        };
        
        Ok((tree, pt_file.procedural_params))
    }
    
    fn convert_tree_to_geometry(tree: &PixelTree) -> PtGeometryData {
        let trunk_segments = tree.trunk.segments.iter().map(|seg| PtTrunkSegment {
            start: [seg.start.x, seg.start.y],
            end: [seg.end.x, seg.end.y],
            width: seg.width,
            color: [101, 67, 33], // Brown trunk color
            texture_coords: None,
        }).collect();
        
        let branches = tree.branches.iter().map(|branch| PtBranch {
            start: [branch.start.x, branch.start.y],
            end: [branch.end.x, branch.end.y],
            thickness: branch.thickness,
            generation: branch.generation,
            parent: branch.parent,
            color: [139, 69, 19], // Saddle brown
            texture_coords: None,
            wind_factor: 1.0 - (branch.generation as f32 * 0.1).min(0.8), // Higher generations sway more
        }).collect();
        
        let leaves = tree.leaves.leaves.iter().map(|leaf| PtLeaf {
            position: [leaf.pos.x, leaf.pos.y],
            rotation: leaf.rot,
            scale: leaf.scale,
            leaf_type: leaf.leaf_type,
            color: leaf.color,
            wind_factor: 1.0,
            season_factor: 1.0,
        }).collect();
        
        let bounding_box = Self::calculate_bounding_box(tree);
        
        PtGeometryData {
            trunk_segments,
            branches,
            leaves,
            bounding_box,
        }
    }
    
    fn convert_geometry_to_tree(
        geometry: PtGeometryData,
        wind_params: &WindParams,
        metadata: &PtMetadata,
    ) -> PixelTree {
        let trunk_segments: Vec<TrunkSegment> = geometry.trunk_segments.iter().map(|seg| {
            TrunkSegment {
                start: Vec2::new(seg.start[0], seg.start[1]),
                end: Vec2::new(seg.end[0], seg.end[1]),
                width: seg.width,
            }
        }).collect();
        
        let trunk_data = TrunkData {
            base_pos: Vec2::ZERO,
            height: trunk_segments.last().map(|s| s.end.y).unwrap_or(100.0),
            base_width: trunk_segments.first().map(|s| s.width).unwrap_or(8.0),
            segments: trunk_segments,
        };
        
        let branches: Vec<Branch> = geometry.branches.iter().map(|branch| {
            Branch {
                start: Vec2::new(branch.start[0], branch.start[1]),
                end: Vec2::new(branch.end[0], branch.end[1]),
                thickness: branch.thickness,
                generation: branch.generation,
                parent: branch.parent,
            }
        }).collect();
        
        let leaves: Vec<Leaf> = geometry.leaves.iter().map(|leaf| {
            Leaf {
                pos: Vec2::new(leaf.position[0], leaf.position[1]),
                rot: leaf.rotation,
                scale: leaf.scale,
                leaf_type: leaf.leaf_type,
                color: leaf.color,
            }
        }).collect();
        
        PixelTree {
            trunk: trunk_data,
            branches,
            leaves: LeafCluster { leaves },
            wind_params: wind_params.clone(),
            lod_level: 0,
            template: metadata.tree_type,
        }
    }
    
    fn extract_materials(_tree: &PixelTree) -> PtMaterialData {
        PtMaterialData {
            trunk_materials: vec![PtMaterial {
                name: "Trunk".to_string(),
                base_color: [101, 67, 33],
                roughness: 0.8,
                metallic: 0.0,
                emission: [0, 0, 0],
                texture_path: None,
            }],
            branch_materials: vec![PtMaterial {
                name: "Branch".to_string(),
                base_color: [139, 69, 19],
                roughness: 0.7,
                metallic: 0.0,
                emission: [0, 0, 0],
                texture_path: None,
            }],
            leaf_materials: vec![PtMaterial {
                name: "Leaf".to_string(),
                base_color: [34, 139, 34],
                roughness: 0.5,
                metallic: 0.0,
                emission: [0, 0, 0],
                texture_path: None,
            }],
            bark_texture: None,
            leaf_texture: None,
        }
    }
    
    fn create_lod_levels(tree: &PixelTree) -> Vec<PtLodLevel> {
        vec![
            PtLodLevel {
                level: 0,
                max_distance: 100.0,
                branch_count: tree.branches.len(),
                leaf_density: 1.0,
                detail_reduction: 0.0,
            },
            PtLodLevel {
                level: 1,
                max_distance: 300.0,
                branch_count: (tree.branches.len() * 3 / 4).max(1),
                leaf_density: 0.7,
                detail_reduction: 0.25,
            },
            PtLodLevel {
                level: 2,
                max_distance: 600.0,
                branch_count: (tree.branches.len() / 2).max(1),
                leaf_density: 0.4,
                detail_reduction: 0.5,
            },
            PtLodLevel {
                level: 3,
                max_distance: f32::MAX,
                branch_count: 0,
                leaf_density: 0.0,
                detail_reduction: 1.0,
            },
        ]
    }
    
    fn calculate_bounding_box(tree: &PixelTree) -> PtBoundingBox {
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        
        // Check trunk segments
        for segment in &tree.trunk.segments {
            min_x = min_x.min(segment.start.x.min(segment.end.x));
            max_x = max_x.max(segment.start.x.max(segment.end.x));
            min_y = min_y.min(segment.start.y.min(segment.end.y));
            max_y = max_y.max(segment.start.y.max(segment.end.y));
        }
        
        // Check branches
        for branch in &tree.branches {
            min_x = min_x.min(branch.start.x.min(branch.end.x));
            max_x = max_x.max(branch.start.x.max(branch.end.x));
            min_y = min_y.min(branch.start.y.min(branch.end.y));
            max_y = max_y.max(branch.start.y.max(branch.end.y));
        }
        
        // Check leaves
        for leaf in &tree.leaves.leaves {
            min_x = min_x.min(leaf.pos.x);
            max_x = max_x.max(leaf.pos.x);
            min_y = min_y.min(leaf.pos.y);
            max_y = max_y.max(leaf.pos.y);
        }
        
        PtBoundingBox {
            min: [min_x, min_y],
            max: [max_x, max_y],
            height: max_y - min_y,
        }
    }
    
    fn default_metadata(tree_type: &TreeTemplate) -> PtMetadata {
        let now = Utc::now();
        PtMetadata {
            tree_name: format!("{:?} Tree", tree_type),
            author: "PixelTree Generator".to_string(),
            created_date: now,
            modified_date: now,
            description: format!("Procedurally generated {:?} tree", tree_type),
            tags: vec!["procedural".to_string(), "pixeltree".to_string()],
            tree_type: *tree_type,
            estimated_age_years: Some(25.0),
            biome: Some("Temperate".to_string()),
            custom_properties: HashMap::new(),
        }
    }

    // ============================================================================
    // VALIDATION FUNCTIONS
    // ============================================================================

    fn validate_path<P: AsRef<Path>>(path: P) -> Result<(), PtFormatError> {
        let path = path.as_ref();
        
        // Check for path traversal attacks
        if path.to_string_lossy().contains("..") {
            return Err(PtFormatError::InvalidPath(
                "Path traversal not allowed".to_string()
            ));
        }
        
        // Check for absolute paths in untrusted contexts
        if path.is_absolute() {
            let path_str = path.to_string_lossy();
            if path_str.starts_with("/dev/") || path_str.starts_with("/proc/") || path_str.starts_with("/sys/") {
                return Err(PtFormatError::InvalidPath(
                    "System directory access not allowed".to_string()
                ));
            }
        }
        
        // Validate extension
        if let Some(ext) = path.extension() {
            if ext != "pt" && ext != "json" {
                return Err(PtFormatError::InvalidPath(
                    format!("Invalid file extension: {}", ext.to_string_lossy())
                ));
            }
        } else {
            return Err(PtFormatError::InvalidPath(
                "Missing file extension".to_string()
            ));
        }
        
        Ok(())
    }

    fn validate_tree(tree: &PixelTree) -> Result<(), PtFormatError> {
        // Validate trunk
        if tree.trunk.segments.len() > MAX_TRUNK_SEGMENTS {
            return Err(PtFormatError::InvalidTreeData(
                format!("Too many trunk segments: {}", tree.trunk.segments.len())
            ));
        }
        
        if tree.trunk.height < 0.0 || tree.trunk.height > 1000.0 {
            return Err(PtFormatError::InvalidTreeData(
                format!("Invalid trunk height: {}", tree.trunk.height)
            ));
        }
        
        if tree.trunk.base_width < 0.0 || tree.trunk.base_width > 100.0 {
            return Err(PtFormatError::InvalidTreeData(
                format!("Invalid trunk width: {}", tree.trunk.base_width)
            ));
        }

        // Validate branches
        if tree.branches.len() > MAX_BRANCHES {
            return Err(PtFormatError::InvalidTreeData(
                format!("Too many branches: {}", tree.branches.len())
            ));
        }

        for (i, branch) in tree.branches.iter().enumerate() {
            if branch.thickness < 0.0 || branch.thickness > 50.0 {
                return Err(PtFormatError::InvalidTreeData(
                    format!("Invalid branch thickness at index {}: {}", i, branch.thickness)
                ));
            }
            
            if branch.generation > 10 {
                return Err(PtFormatError::InvalidTreeData(
                    format!("Invalid branch generation at index {}: {}", i, branch.generation)
                ));
            }

            // Validate positions are reasonable
            let start_mag = branch.start.length();
            let end_mag = branch.end.length();
            if start_mag > 2000.0 || end_mag > 2000.0 {
                return Err(PtFormatError::InvalidTreeData(
                    format!("Branch position too far from origin at index {}", i)
                ));
            }
        }

        // Validate leaves
        if tree.leaves.leaves.len() > MAX_LEAVES {
            return Err(PtFormatError::InvalidTreeData(
                format!("Too many leaves: {}", tree.leaves.leaves.len())
            ));
        }

        for (i, leaf) in tree.leaves.leaves.iter().enumerate() {
            if leaf.scale < 0.0 || leaf.scale > 10.0 {
                return Err(PtFormatError::InvalidTreeData(
                    format!("Invalid leaf scale at index {}: {}", i, leaf.scale)
                ));
            }

            let pos_mag = leaf.pos.length();
            if pos_mag > 2000.0 {
                return Err(PtFormatError::InvalidTreeData(
                    format!("Leaf position too far from origin at index {}", i)
                ));
            }
        }

        // Validate wind parameters
        if tree.wind_params.strength < 0.0 || tree.wind_params.strength > 100.0 {
            return Err(PtFormatError::InvalidTreeData(
                format!("Invalid wind strength: {}", tree.wind_params.strength)
            ));
        }

        Ok(())
    }

    fn validate_params(params: &GenerationParams) -> Result<(), PtFormatError> {
        if params.height_range.0 < 0.0 || params.height_range.1 < params.height_range.0 || params.height_range.1 > 2000.0 {
            return Err(PtFormatError::InvalidTreeData(
                format!("Invalid height range: {:?}", params.height_range)
            ));
        }

        if params.trunk_width_range.0 < 0.0 || params.trunk_width_range.1 < params.trunk_width_range.0 || params.trunk_width_range.1 > 200.0 {
            return Err(PtFormatError::InvalidTreeData(
                format!("Invalid trunk width range: {:?}", params.trunk_width_range)
            ));
        }

        if params.branch_angle_variance < 0.0 || params.branch_angle_variance > 360.0 {
            return Err(PtFormatError::InvalidTreeData(
                format!("Invalid branch angle variance: {}", params.branch_angle_variance)
            ));
        }

        if params.leaf_density < 0.0 || params.leaf_density > 100.0 {
            return Err(PtFormatError::InvalidTreeData(
                format!("Invalid leaf density: {}", params.leaf_density)
            ));
        }

        Ok(())
    }

    fn validate_and_sanitize_metadata(metadata: &mut PtMetadata) -> Result<(), PtFormatError> {
        // Sanitize string lengths
        if metadata.tree_name.len() > MAX_TREE_NAME_LEN {
            metadata.tree_name.truncate(MAX_TREE_NAME_LEN);
        }

        if metadata.author.len() > MAX_AUTHOR_LEN {
            metadata.author.truncate(MAX_AUTHOR_LEN);
        }

        if metadata.description.len() > MAX_DESCRIPTION_LEN {
            metadata.description.truncate(MAX_DESCRIPTION_LEN);
        }

        // Validate and limit custom properties
        if metadata.custom_properties.len() > MAX_CUSTOM_PROPERTIES {
            return Err(PtFormatError::InvalidTreeData(
                format!("Too many custom properties: {}", metadata.custom_properties.len())
            ));
        }

        let mut sanitized_props = HashMap::new();
        for (key, value) in &metadata.custom_properties {
            if key.len() > MAX_PROPERTY_KEY_LEN || value.len() > MAX_PROPERTY_VALUE_LEN {
                continue; // Skip oversized properties
            }
            
            // Sanitize key/value (remove control characters)
            let clean_key: String = key.chars().filter(|c| !c.is_control()).collect();
            let clean_value: String = value.chars().filter(|c| !c.is_control()).collect();
            
            if !clean_key.is_empty() && !clean_value.is_empty() {
                sanitized_props.insert(clean_key, clean_value);
            }
        }
        metadata.custom_properties = sanitized_props;

        // Validate age if present
        if let Some(age) = metadata.estimated_age_years {
            if age < 0.0 || age > 10000.0 {
                return Err(PtFormatError::InvalidTreeData(
                    format!("Invalid tree age: {}", age)
                ));
            }
        }

        Ok(())
    }

    fn validate_file_header(header: &PtFileHeader) -> Result<(), PtFormatError> {
        if header.magic != "PIXELTREE" {
            return Err(PtFormatError::InvalidHeader(format!(
                "Expected 'PIXELTREE', got '{}'", header.magic
            )));
        }

        if header.format_version == 0 || header.format_version > 10 {
            return Err(PtFormatError::UnsupportedVersion(format!(
                "Invalid format version: {}", header.format_version
            )));
        }

        if header.format_version > 1 {
            return Err(PtFormatError::UnsupportedVersion(format!(
                "Format version {} not supported, maximum supported is 1", header.format_version
            )));
        }

        Ok(())
    }

    fn validate_loaded_data(pt_file: &PixelTreeFile) -> Result<(), PtFormatError> {
        // Validate metadata
        if pt_file.metadata.tree_name.is_empty() {
            return Err(PtFormatError::DataCorruption("Empty tree name".to_string()));
        }

        // Validate geometry data if present
        if let Some(ref geometry) = pt_file.geometry_data {
            if geometry.trunk_segments.len() > MAX_TRUNK_SEGMENTS {
                return Err(PtFormatError::DataCorruption(
                    format!("Too many trunk segments in file: {}", geometry.trunk_segments.len())
                ));
            }

            if geometry.branches.len() > MAX_BRANCHES {
                return Err(PtFormatError::DataCorruption(
                    format!("Too many branches in file: {}", geometry.branches.len())
                ));
            }

            if geometry.leaves.len() > MAX_LEAVES {
                return Err(PtFormatError::DataCorruption(
                    format!("Too many leaves in file: {}", geometry.leaves.len())
                ));
            }
        }

        // Validate LOD levels
        if pt_file.lod_levels.is_empty() || pt_file.lod_levels.len() > 10 {
            return Err(PtFormatError::DataCorruption(
                format!("Invalid number of LOD levels: {}", pt_file.lod_levels.len())
            ));
        }

        for (i, lod) in pt_file.lod_levels.iter().enumerate() {
            if lod.max_distance < 0.0 {
                return Err(PtFormatError::DataCorruption(
                    format!("Invalid LOD distance at level {}: {}", i, lod.max_distance)
                ));
            }
            
            if lod.leaf_density < 0.0 || lod.leaf_density > 10.0 {
                return Err(PtFormatError::DataCorruption(
                    format!("Invalid LOD leaf density at level {}: {}", i, lod.leaf_density)
                ));
            }
        }

        Ok(())
    }
}

// ============================================================================
// PIXELTREE FOREST FORMAT (.ptf)
// ============================================================================

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtfForestFile {
    pub header: PtfFileHeader,
    pub forest_metadata: PtfForestMetadata,
    pub tree_groups: Vec<PtfTreeGroup>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtfFileHeader {
    pub format_version: String, // "PixelTree Forest v1.0"
    pub magic: String, // "PTFOREST"
    pub encoding: PtfEncoding,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PtfEncoding {
    Text, // Human-readable text format (like STF)
    Binary, // Compact binary format
    Json, // JSON format for tooling
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtfForestMetadata {
    pub forest_name: String,
    pub author: String,
    pub created_date: DateTime<Utc>,
    pub modified_date: DateTime<Utc>,
    pub biome: String, // "Old Growth Forest", "Pine Grove", etc.
    pub season: PtfSeason,
    pub time_of_day: Option<PtfTimeOfDay>,
    pub global_wind: WindParams,
    pub terrain: Option<PtfTerrain>,
    pub lighting: Option<PtfLighting>,
    pub weather: Option<PtfWeather>,
    pub total_trees: usize,
    pub bounding_box: PtBoundingBox,
    pub tags: Vec<String>,
    pub custom_properties: HashMap<String, String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PtfSeason {
    Spring,
    Summer,
    Autumn,
    Winter,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtfTimeOfDay {
    pub hour: f32, // 0.0-23.99
    pub ambient_light: f32, // 0.0-1.0
    pub sun_angle: f32, // degrees
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtfTerrain {
    pub elevation_min: f32,
    pub elevation_max: f32,
    pub slope_factor: f32, // 0.0-1.0
    pub soil_quality_base: f32, // 0.0-1.0
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtfLighting {
    pub ambient_color: [f32; 3], // RGB
    pub sun_color: [f32; 3], // RGB
    pub shadow_intensity: f32, // 0.0-1.0
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtfWeather {
    pub condition: PtfWeatherCondition,
    pub intensity: f32, // 0.0-1.0
    pub wind_modifier: f32, // Multiplier for global wind
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PtfWeatherCondition {
    Clear,
    Cloudy,
    Rainy,
    Stormy,
    Foggy,
    Snowy,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtfTreeGroup {
    pub tree_file: String, // Path to .pt file
    pub instances: Vec<PtfTreeInstance>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PtfTreeInstance {
    pub position: Vec3,
    pub rotation: f32,
    pub scale: f32,
    pub health_factor: f32, // 0.0-1.0, affects appearance
    pub tags: Vec<String>, // ["healthy", "mature", "diseased", etc.]
    pub custom_properties: HashMap<String, String>,
}

#[derive(Error, Debug)]
pub enum PtfFormatError {
    #[error("Invalid PTF header: {0}")]
    InvalidHeader(String),
    #[error("Unsupported PTF version: {0}")]
    UnsupportedVersion(String),
    #[error("PTF parsing error: {0}")]
    ParseError(String),
    #[error("Tree file not found: {0}")]
    TreeFileNotFound(String),
    #[error("Invalid tree instance data: {0}")]
    InvalidInstanceData(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

// ============================================================================
// PTF FORMAT IMPLEMENTATION
// ============================================================================

pub struct PtfForestManager;

impl PtfForestManager {
    pub fn save_forest<P: AsRef<Path>>(
        forest_metadata: &PtfForestMetadata,
        tree_groups: &[(String, Vec<PtfTreeInstance>)],
        path: P,
        encoding: PtfEncoding,
    ) -> Result<(), PtfFormatError> {
        // Validate input parameters
        Self::validate_forest_for_save(forest_metadata, tree_groups)?;
        Self::validate_ptf_path(path.as_ref())?;

        let tree_groups: Vec<PtfTreeGroup> = tree_groups
            .iter()
            .map(|(tree_file, instances)| PtfTreeGroup {
                tree_file: tree_file.clone(),
                instances: instances.clone(),
            })
            .collect();

        let ptf_forest = PtfForestFile {
            header: PtfFileHeader {
                format_version: "PixelTree Forest v1.0".to_string(),
                magic: "PTFOREST".to_string(),
                encoding: encoding.clone(),
            },
            forest_metadata: forest_metadata.clone(),
            tree_groups,
        };

        let content = match encoding {
            PtfEncoding::Text => Self::serialize_to_text(&ptf_forest)?,
            PtfEncoding::Binary => bincode::serialize(&ptf_forest)
                .map_err(|e| PtfFormatError::SerializationError(e.to_string()))?,
            PtfEncoding::Json => serde_json::to_vec_pretty(&ptf_forest)
                .map_err(|e| PtfFormatError::SerializationError(e.to_string()))?,
        };

        // Validate serialized size
        if content.len() as u64 > MAX_PTF_FILE_SIZE {
            return Err(PtfFormatError::SerializationError(
                format!("Serialized PTF file size {} exceeds maximum {}", content.len(), MAX_PTF_FILE_SIZE)
            ));
        }

        // Atomic write - write to temporary file first, then rename
        let temp_path = path.as_ref().with_extension("ptf.tmp");
        std::fs::write(&temp_path, &content).map_err(|e| {
            let _ = std::fs::remove_file(&temp_path);
            PtfFormatError::IoError(e)
        })?;
        
        // Atomic rename
        std::fs::rename(&temp_path, path.as_ref()).map_err(|e| {
            let _ = std::fs::remove_file(&temp_path);
            PtfFormatError::IoError(e)
        })?;

        Ok(())
    }

    pub fn load_forest<P: AsRef<Path>>(path: P) -> Result<PtfForestFile, PtfFormatError> {
        // Security validation
        Self::validate_ptf_path(path.as_ref())?;
        
        // Check file size before reading
        let metadata = std::fs::metadata(path.as_ref())?;
        let file_size = metadata.len();
        
        if file_size > MAX_PTF_FILE_SIZE {
            return Err(PtfFormatError::SerializationError(
                format!("PTF file size {} exceeds maximum {}", file_size, MAX_PTF_FILE_SIZE)
            ));
        }
        
        if file_size < MIN_PTF_FILE_SIZE {
            return Err(PtfFormatError::SerializationError(
                format!("PTF file size {} is below minimum {}", file_size, MIN_PTF_FILE_SIZE)
            ));
        }

        let content = std::fs::read(path.as_ref())?;

        // Verify content length matches file size
        if content.len() as u64 != file_size {
            return Err(PtfFormatError::SerializationError(
                "File size mismatch - possible corruption".to_string()
            ));
        }

        let ptf_forest = {
            // Try to detect format by content
            if let Ok(text) = std::str::from_utf8(&content) {
                if text.starts_with("# PixelTree Forest") {
                    Self::parse_text_format(text)?
                } else if text.trim_start().starts_with('{') {
                    serde_json::from_slice(&content)
                        .map_err(|e| PtfFormatError::SerializationError(e.to_string()))?
                } else {
                    return Err(PtfFormatError::SerializationError(
                        "Unrecognized text format".to_string()
                    ));
                }
            } else {
                // Try binary format
                bincode::deserialize(&content)
                    .map_err(|e| PtfFormatError::SerializationError(e.to_string()))?
            }
        };

        // Validate loaded data
        Self::validate_loaded_forest(&ptf_forest)?;
        
        Ok(ptf_forest)
    }

    pub fn spawn_forest_from_ptf<P: AsRef<Path>>(
        commands: &mut Commands,
        ptf_path: P,
        spatial_index: &mut TreeSpatialIndex,
    ) -> Result<Vec<Entity>, PtfFormatError> {
        let base_path = ptf_path.as_ref().parent().unwrap_or(Path::new("."));
        let ptf_forest = Self::load_forest(&ptf_path)?;
        let mut entities = Vec::new();

        for tree_group in &ptf_forest.tree_groups {
            let tree_file_path = base_path.join(&tree_group.tree_file);
            
            // Load the base tree template
            let (base_tree, _base_params) = PtFileManager::load_tree(&tree_file_path)
                .map_err(|_| PtfFormatError::TreeFileNotFound(tree_group.tree_file.clone()))?;

            for instance in &tree_group.instances {
                // Clone and modify tree based on instance properties
                let mut tree = base_tree.clone();
                Self::apply_instance_modifications(&mut tree, instance, &ptf_forest.forest_metadata);

                let entity = commands.spawn((
                    Transform::from_translation(instance.position)
                        .with_rotation(Quat::from_rotation_z(instance.rotation))
                        .with_scale(Vec3::splat(instance.scale)),
                    GlobalTransform::default(),
                    tree,
                    Visibility::default(),
                    InheritedVisibility::default(),
                    ViewVisibility::default(),
                )).id();

                spatial_index.insert(entity, instance.position.truncate());
                entities.push(entity);
            }
        }

        Ok(entities)
    }

    // ============================================================================
    // PTF VALIDATION FUNCTIONS
    // ============================================================================

    fn validate_ptf_path(path: &Path) -> Result<(), PtfFormatError> {
        // Check for path traversal attempts
        let path_str = path.to_string_lossy();
        if path_str.contains("..") || path_str.contains('\0') {
            return Err(PtfFormatError::SerializationError(
                "Invalid path: contains path traversal or null bytes".to_string()
            ));
        }

        // Check path length
        if path_str.len() > MAX_TREE_FILE_PATH_LENGTH {
            return Err(PtfFormatError::SerializationError(
                format!("Path too long: {} > {}", path_str.len(), MAX_TREE_FILE_PATH_LENGTH)
            ));
        }

        Ok(())
    }

    fn validate_forest_for_save(
        metadata: &PtfForestMetadata,
        tree_groups: &[(String, Vec<PtfTreeInstance>)]
    ) -> Result<(), PtfFormatError> {
        // Validate metadata
        Self::validate_forest_metadata(metadata)?;

        // Validate tree groups structure
        if tree_groups.len() > MAX_TREE_GROUPS {
            return Err(PtfFormatError::SerializationError(
                format!("Too many tree groups: {} > {}", tree_groups.len(), MAX_TREE_GROUPS)
            ));
        }

        let mut total_instances = 0;
        for (tree_file, instances) in tree_groups {
            // Validate tree file path
            if tree_file.len() > MAX_TREE_FILE_PATH_LENGTH {
                return Err(PtfFormatError::SerializationError(
                    format!("Tree file path too long: {}", tree_file.len())
                ));
            }

            // Validate instances count
            if instances.len() > MAX_INSTANCES_PER_GROUP {
                return Err(PtfFormatError::SerializationError(
                    format!("Too many instances in group: {} > {}", instances.len(), MAX_INSTANCES_PER_GROUP)
                ));
            }

            total_instances += instances.len();
            if total_instances > MAX_TREE_INSTANCES {
                return Err(PtfFormatError::SerializationError(
                    format!("Total instances exceed maximum: {} > {}", total_instances, MAX_TREE_INSTANCES)
                ));
            }

            // Validate each instance
            for instance in instances {
                Self::validate_tree_instance(instance)?;
            }
        }

        Ok(())
    }

    fn validate_forest_metadata(metadata: &PtfForestMetadata) -> Result<(), PtfFormatError> {
        // Check string lengths
        if metadata.forest_name.len() > MAX_TREE_NAME_LEN {
            return Err(PtfFormatError::SerializationError(
                format!("Forest name too long: {}", metadata.forest_name.len())
            ));
        }

        if metadata.author.len() > MAX_AUTHOR_LEN {
            return Err(PtfFormatError::SerializationError(
                format!("Author name too long: {}", metadata.author.len())
            ));
        }

        if metadata.biome.len() > MAX_DESCRIPTION_LEN {
            return Err(PtfFormatError::SerializationError(
                format!("Biome description too long: {}", metadata.biome.len())
            ));
        }

        // Validate tags count
        if metadata.tags.len() > MAX_PTF_TAGS {
            return Err(PtfFormatError::SerializationError(
                format!("Too many tags: {} > {}", metadata.tags.len(), MAX_PTF_TAGS)
            ));
        }

        // Validate custom properties count
        if metadata.custom_properties.len() > MAX_PTF_CUSTOM_PROPERTIES {
            return Err(PtfFormatError::SerializationError(
                format!("Too many custom properties: {} > {}", metadata.custom_properties.len(), MAX_PTF_CUSTOM_PROPERTIES)
            ));
        }

        // Validate custom property keys and values
        for (key, value) in &metadata.custom_properties {
            if key.len() > MAX_PROPERTY_KEY_LEN {
                return Err(PtfFormatError::SerializationError(
                    format!("Property key too long: {} > {}", key.len(), MAX_PROPERTY_KEY_LEN)
                ));
            }
            if value.len() > MAX_PROPERTY_VALUE_LEN {
                return Err(PtfFormatError::SerializationError(
                    format!("Property value too long: {} > {}", value.len(), MAX_PROPERTY_VALUE_LEN)
                ));
            }
        }

        // Validate bounding box
        if metadata.bounding_box.min[0] > metadata.bounding_box.max[0] ||
           metadata.bounding_box.min[1] > metadata.bounding_box.max[1] {
            return Err(PtfFormatError::SerializationError(
                "Invalid bounding box: min > max".to_string()
            ));
        }

        if metadata.bounding_box.height <= 0.0 || metadata.bounding_box.height > 10000.0 {
            return Err(PtfFormatError::SerializationError(
                format!("Invalid bounding box height: {}", metadata.bounding_box.height)
            ));
        }

        Ok(())
    }

    fn validate_tree_instance(instance: &PtfTreeInstance) -> Result<(), PtfFormatError> {
        // Validate scale bounds
        if instance.scale <= 0.0 || instance.scale > 100.0 {
            return Err(PtfFormatError::SerializationError(
                format!("Invalid tree scale: {}", instance.scale)
            ));
        }

        // Validate health factor
        if instance.health_factor < 0.0 || instance.health_factor > 1.0 {
            return Err(PtfFormatError::SerializationError(
                format!("Invalid health factor: {}", instance.health_factor)
            ));
        }

        // Validate position bounds (prevent extreme values)
        let pos_limit = 1000000.0;
        if instance.position.x.abs() > pos_limit || 
           instance.position.y.abs() > pos_limit ||
           instance.position.z.abs() > pos_limit {
            return Err(PtfFormatError::SerializationError(
                "Tree position exceeds reasonable bounds".to_string()
            ));
        }

        // Validate tags
        if instance.tags.len() > MAX_PTF_TAGS {
            return Err(PtfFormatError::SerializationError(
                format!("Too many instance tags: {} > {}", instance.tags.len(), MAX_PTF_TAGS)
            ));
        }

        // Validate custom properties
        if instance.custom_properties.len() > MAX_CUSTOM_PROPERTIES {
            return Err(PtfFormatError::SerializationError(
                format!("Too many instance properties: {} > {}", instance.custom_properties.len(), MAX_CUSTOM_PROPERTIES)
            ));
        }

        Ok(())
    }

    fn validate_loaded_forest(forest: &PtfForestFile) -> Result<(), PtfFormatError> {
        // Validate header
        if forest.header.magic != "PTFOREST" {
            return Err(PtfFormatError::SerializationError(
                "Invalid magic header".to_string()
            ));
        }

        // Validate metadata
        Self::validate_forest_metadata(&forest.forest_metadata)?;

        // Validate tree groups
        if forest.tree_groups.len() > MAX_TREE_GROUPS {
            return Err(PtfFormatError::SerializationError(
                format!("Too many tree groups: {} > {}", forest.tree_groups.len(), MAX_TREE_GROUPS)
            ));
        }

        let mut total_instances = 0;
        for group in &forest.tree_groups {
            if group.instances.len() > MAX_INSTANCES_PER_GROUP {
                return Err(PtfFormatError::SerializationError(
                    format!("Too many instances in group: {} > {}", group.instances.len(), MAX_INSTANCES_PER_GROUP)
                ));
            }

            total_instances += group.instances.len();
            if total_instances > MAX_TREE_INSTANCES {
                return Err(PtfFormatError::SerializationError(
                    format!("Total instances exceed maximum: {} > {}", total_instances, MAX_TREE_INSTANCES)
                ));
            }

            for instance in &group.instances {
                Self::validate_tree_instance(instance)?;
            }
        }

        Ok(())
    }

    fn serialize_to_text(forest: &PtfForestFile) -> Result<Vec<u8>, PtfFormatError> {
        let mut content = String::new();
        
        // Header comments
        content.push_str(&format!("# PixelTree Forest v1.0\n"));
        content.push_str(&format!("# Forest: {}\n", forest.forest_metadata.forest_name));
        content.push_str(&format!("# Author: {}\n", forest.forest_metadata.author));
        content.push_str(&format!("# Created: {}\n", forest.forest_metadata.created_date.format("%Y-%m-%dT%H:%M:%SZ")));
        content.push_str(&format!("# Biome: {}\n", forest.forest_metadata.biome));
        content.push_str(&format!("# Season: {:?}\n", forest.forest_metadata.season));
        content.push_str(&format!("# Wind: Global(strength={:.1}, direction=[{:.1},{:.1}])\n", 
            forest.forest_metadata.global_wind.strength,
            forest.forest_metadata.global_wind.direction.x,
            forest.forest_metadata.global_wind.direction.y));
        
        if let Some(time) = &forest.forest_metadata.time_of_day {
            content.push_str(&format!("# Lighting: TimeOfDay(hour={:.1}, ambient={:.1})\n", 
                time.hour, time.ambient_light));
        }
        
        if let Some(terrain) = &forest.forest_metadata.terrain {
            content.push_str(&format!("# Terrain: Elevation(min={:.1}, max={:.1}, slope={:.1})\n",
                terrain.elevation_min, terrain.elevation_max, terrain.slope_factor));
        }
        
        content.push_str("\n");

        // Tree groups
        for tree_group in &forest.tree_groups {
            content.push_str(&format!("{}\n", tree_group.tree_file));
            content.push_str(&format!("{}\n", tree_group.instances.len()));
            
            for instance in &tree_group.instances {
                content.push_str(&format!(
                    "{:.1} {:.1} {:.1} {:.6} {:.2} {:.2}",
                    instance.position.x, instance.position.y, instance.position.z,
                    instance.rotation, instance.scale, instance.health_factor
                ));
                
                if !instance.tags.is_empty() {
                    content.push_str(&format!(" [{}]", instance.tags.join(",")));
                }
                
                if !instance.custom_properties.is_empty() {
                    let props: Vec<String> = instance.custom_properties
                        .iter()
                        .map(|(k, v)| format!("{}:{}", k, v))
                        .collect();
                    content.push_str(&format!(" {{{}}}", props.join(",")));
                }
                
                content.push_str("\n");
            }
            content.push_str("\n");
        }

        Ok(content.into_bytes())
    }

    fn parse_text_format(content: &str) -> Result<PtfForestFile, PtfFormatError> {
        let lines: Vec<&str> = content.lines().collect();
        let mut line_idx = 0;
        
        // Parse header comments
        let mut forest_name = "Unnamed Forest".to_string();
        let mut author = "Unknown".to_string();
        let created_date = Utc::now();
        let mut biome = "Temperate".to_string();
        let mut season = PtfSeason::Summer;
        let global_wind = WindParams::default();
        let time_of_day = None;
        let terrain = None;
        
        while line_idx < lines.len() && lines[line_idx].starts_with('#') {
            let line = lines[line_idx].trim_start_matches('#').trim();
            
            if let Some(name) = line.strip_prefix("Forest: ") {
                forest_name = name.to_string();
            } else if let Some(auth) = line.strip_prefix("Author: ") {
                author = auth.to_string();
            } else if let Some(biome_str) = line.strip_prefix("Biome: ") {
                biome = biome_str.to_string();
            } else if let Some(season_str) = line.strip_prefix("Season: ") {
                season = match season_str {
                    "Spring" => PtfSeason::Spring,
                    "Summer" => PtfSeason::Summer,
                    "Autumn" => PtfSeason::Autumn,
                    "Winter" => PtfSeason::Winter,
                    _ => PtfSeason::Summer,
                };
            }
            line_idx += 1;
        }

        // Skip empty lines
        while line_idx < lines.len() && lines[line_idx].trim().is_empty() {
            line_idx += 1;
        }

        let mut tree_groups = Vec::new();
        let mut total_trees = 0;

        // Parse tree groups
        while line_idx < lines.len() {
            if lines[line_idx].trim().is_empty() {
                line_idx += 1;
                continue;
            }

            let tree_file = lines[line_idx].trim().to_string();
            line_idx += 1;

            if line_idx >= lines.len() {
                break;
            }

            let instance_count: usize = lines[line_idx].trim().parse()
                .map_err(|_| PtfFormatError::ParseError(format!("Invalid instance count: {}", lines[line_idx])))?;
            line_idx += 1;
            total_trees += instance_count;

            let mut instances = Vec::with_capacity(instance_count);

            for _ in 0..instance_count {
                if line_idx >= lines.len() {
                    return Err(PtfFormatError::ParseError("Unexpected end of file".to_string()));
                }

                let instance = Self::parse_instance_line(lines[line_idx])?;
                instances.push(instance);
                line_idx += 1;
            }

            tree_groups.push(PtfTreeGroup {
                tree_file,
                instances,
            });
        }

        let forest_metadata = PtfForestMetadata {
            forest_name,
            author,
            created_date,
            modified_date: Utc::now(),
            biome,
            season,
            time_of_day,
            global_wind,
            terrain,
            lighting: None,
            weather: None,
            total_trees,
            bounding_box: PtBoundingBox {
                min: [-1000.0, -1000.0],
                max: [1000.0, 1000.0],
                height: 200.0,
            },
            tags: Vec::new(),
            custom_properties: HashMap::new(),
        };

        Ok(PtfForestFile {
            header: PtfFileHeader {
                format_version: "PixelTree Forest v1.0".to_string(),
                magic: "PTFOREST".to_string(),
                encoding: PtfEncoding::Text,
            },
            forest_metadata,
            tree_groups,
        })
    }

    fn parse_instance_line(line: &str) -> Result<PtfTreeInstance, PtfFormatError> {
        let line = line.trim();
        
        // Split into main parts and optional parts
        let (coords_part, rest) = if let Some(bracket_pos) = line.find('[') {
            line.split_at(bracket_pos)
        } else if let Some(brace_pos) = line.find('{') {
            line.split_at(brace_pos)
        } else {
            (line, "")
        };

        // Parse coordinates: x y z rotation scale health_factor
        let coords: Vec<&str> = coords_part.trim().split_whitespace().collect();
        if coords.len() < 5 {
            return Err(PtfFormatError::InvalidInstanceData(
                format!("Expected at least 5 coordinates, got {}", coords.len())
            ));
        }

        let x = coords[0].parse::<f32>()
            .map_err(|_| PtfFormatError::InvalidInstanceData(format!("Invalid x: {}", coords[0])))?;
        let y = coords[1].parse::<f32>()
            .map_err(|_| PtfFormatError::InvalidInstanceData(format!("Invalid y: {}", coords[1])))?;
        let z = coords[2].parse::<f32>()
            .map_err(|_| PtfFormatError::InvalidInstanceData(format!("Invalid z: {}", coords[2])))?;
        let rotation = coords[3].parse::<f32>()
            .map_err(|_| PtfFormatError::InvalidInstanceData(format!("Invalid rotation: {}", coords[3])))?;
        let scale = coords[4].parse::<f32>()
            .map_err(|_| PtfFormatError::InvalidInstanceData(format!("Invalid scale: {}", coords[4])))?;
        
        let health_factor = if coords.len() > 5 {
            coords[5].parse::<f32>()
                .map_err(|_| PtfFormatError::InvalidInstanceData(format!("Invalid health_factor: {}", coords[5])))?
        } else {
            1.0
        };

        // Parse tags [tag1,tag2,tag3]
        let mut tags = Vec::new();
        if let Some(start) = rest.find('[') {
            if let Some(end) = rest[start..].find(']') {
                let tags_str = &rest[start+1..start+end];
                tags = tags_str.split(',').map(|s| s.trim().to_string()).collect();
            }
        }

        // Parse custom properties {key1:value1,key2:value2}
        let mut custom_properties = HashMap::new();
        if let Some(start) = rest.find('{') {
            if let Some(end) = rest[start..].find('}') {
                let props_str = &rest[start+1..start+end];
                for prop in props_str.split(',') {
                    if let Some((key, value)) = prop.split_once(':') {
                        custom_properties.insert(key.trim().to_string(), value.trim().to_string());
                    }
                }
            }
        }

        Ok(PtfTreeInstance {
            position: Vec3::new(x, y, z),
            rotation,
            scale,
            health_factor,
            tags,
            custom_properties,
        })
    }

    fn apply_instance_modifications(
        tree: &mut PixelTree,
        instance: &PtfTreeInstance,
        forest_metadata: &PtfForestMetadata,
    ) {
        // Apply health factor to leaf density and color
        if instance.health_factor < 1.0 {
            let health_reduction = 1.0 - instance.health_factor;
            
            // Reduce leaf count based on health
            let target_leaf_count = (tree.leaves.leaves.len() as f32 * instance.health_factor) as usize;
            tree.leaves.leaves.truncate(target_leaf_count);
            
            // Modify leaf colors for diseased/unhealthy trees
            for leaf in &mut tree.leaves.leaves {
                if health_reduction > 0.3 {
                    // Brown/yellow diseased leaves
                    leaf.color[0] = (leaf.color[0] as f32 * 0.6 + 139.0 * 0.4) as u8; // More brown
                    leaf.color[1] = (leaf.color[1] as f32 * 0.8 + 69.0 * 0.2) as u8;  // Less green
                    leaf.color[2] = (leaf.color[2] as f32 * 0.4 + 19.0 * 0.6) as u8;  // Less blue
                }
            }
        }

        // Apply seasonal modifications
        match forest_metadata.season {
            PtfSeason::Autumn => {
                for leaf in &mut tree.leaves.leaves {
                    // Autumn colors - reds, oranges, yellows
                    match leaf.leaf_type % 3 {
                        0 => { leaf.color = [255, 69, 0]; }   // Orange red
                        1 => { leaf.color = [255, 215, 0]; }  // Gold
                        2 => { leaf.color = [205, 92, 92]; }  // Indian red
                        _ => {}
                    }
                }
            },
            PtfSeason::Winter => {
                // Remove most leaves for winter (keep only 10%)
                let winter_leaf_count = (tree.leaves.leaves.len() / 10).max(1);
                tree.leaves.leaves.truncate(winter_leaf_count);
            },
            PtfSeason::Spring => {
                // Brighter, lighter greens
                for leaf in &mut tree.leaves.leaves {
                    leaf.color = [144, 238, 144]; // Light green
                }
            },
            PtfSeason::Summer => {
                // Default rich greens - no modification needed
            }
        }

        // Apply custom property modifications
        if let Some(age_str) = instance.custom_properties.get("age_years") {
            if let Ok(age) = age_str.parse::<f32>() {
                if age > 50.0 {
                    // Ancient trees - more gnarled appearance
                    tree.trunk.base_width *= 1.2;
                } else if age < 15.0 {
                    // Young trees - thinner trunk, fewer branches
                    tree.trunk.base_width *= 0.7;
                    let young_branch_count = (tree.branches.len() * 2 / 3).max(1);
                    tree.branches.truncate(young_branch_count);
                }
            }
        }

        // Apply wind modifications
        tree.wind_params = forest_metadata.global_wind.clone();
        
        // Modify wind response based on tags
        if instance.tags.contains(&"ancient".to_string()) {
            tree.wind_params.strength *= 0.7; // Ancient trees sway less
        } else if instance.tags.contains(&"young".to_string()) {
            tree.wind_params.strength *= 1.3; // Young trees sway more
        }
    }

    pub fn create_sample_ptf() -> PtfForestFile {
        PtfForestFile {
            header: PtfFileHeader {
                format_version: "PixelTree Forest v1.0".to_string(),
                magic: "PTFOREST".to_string(),
                encoding: PtfEncoding::Text,
            },
            forest_metadata: PtfForestMetadata {
                forest_name: "Ancient Grove".to_string(),
                author: "PixelTree Demo".to_string(),
                created_date: Utc::now(),
                modified_date: Utc::now(),
                biome: "Old Growth Forest".to_string(),
                season: PtfSeason::Summer,
                time_of_day: Some(PtfTimeOfDay {
                    hour: 14.0,
                    ambient_light: 0.7,
                    sun_angle: 45.0,
                }),
                global_wind: WindParams {
                    strength: 2.0,
                    frequency: 1.0,
                    turbulence: 0.3,
                    direction: Vec2::new(1.0, 0.3),
                },
                terrain: Some(PtfTerrain {
                    elevation_min: 0.0,
                    elevation_max: 50.0,
                    slope_factor: 0.1,
                    soil_quality_base: 0.8,
                }),
                lighting: Some(PtfLighting {
                    ambient_color: [0.7, 0.7, 0.8],
                    sun_color: [1.0, 0.9, 0.7],
                    shadow_intensity: 0.6,
                }),
                weather: Some(PtfWeather {
                    condition: PtfWeatherCondition::Clear,
                    intensity: 0.0,
                    wind_modifier: 1.0,
                }),
                total_trees: 8,
                bounding_box: PtBoundingBox {
                    min: [-150.0, -100.0],
                    max: [150.0, 200.0],
                    height: 180.0,
                },
                tags: vec!["demo".to_string(), "ancient_grove".to_string()],
                custom_properties: {
                    let mut props = HashMap::new();
                    props.insert("biome_type".to_string(), "old_growth".to_string());
                    props.insert("conservation_status".to_string(), "protected".to_string());
                    props
                },
            },
            tree_groups: vec![
                PtfTreeGroup {
                    tree_file: "sample_Oak_tree.pt".to_string(),
                    instances: vec![
                        PtfTreeInstance {
                            position: Vec3::new(-100.0, 0.0, 0.0),
                            rotation: 0.0,
                            scale: 1.2,
                            health_factor: 1.0,
                            tags: vec!["healthy".to_string(), "mature".to_string()],
                            custom_properties: {
                                let mut props = HashMap::new();
                                props.insert("age_years".to_string(), "45".to_string());
                                props.insert("soil_quality".to_string(), "0.8".to_string());
                                props
                            },
                        },
                        PtfTreeInstance {
                            position: Vec3::new(-50.0, 0.0, 0.0),
                            rotation: 0.523599,
                            scale: 0.9,
                            health_factor: 0.8,
                            tags: vec!["healthy".to_string()],
                            custom_properties: {
                                let mut props = HashMap::new();
                                props.insert("age_years".to_string(), "30".to_string());
                                props.insert("soil_quality".to_string(), "0.6".to_string());
                                props
                            },
                        },
                        PtfTreeInstance {
                            position: Vec3::new(0.0, 80.0, 0.0),
                            rotation: 0.0,
                            scale: 1.3,
                            health_factor: 1.0,
                            tags: vec!["ancient".to_string(), "patriarch".to_string()],
                            custom_properties: {
                                let mut props = HashMap::new();
                                props.insert("age_years".to_string(), "100".to_string());
                                props.insert("soil_quality".to_string(), "1.0".to_string());
                                props.insert("landmark".to_string(), "true".to_string());
                                props
                            },
                        },
                    ],
                },
                PtfTreeGroup {
                    tree_file: "sample_Pine_tree.pt".to_string(),
                    instances: vec![
                        PtfTreeInstance {
                            position: Vec3::new(-80.0, 120.0, 0.0),
                            rotation: 0.785398,
                            scale: 0.8,
                            health_factor: 0.7,
                            tags: vec!["young".to_string()],
                            custom_properties: {
                                let mut props = HashMap::new();
                                props.insert("age_years".to_string(), "15".to_string());
                                props.insert("elevation".to_string(), "20.0".to_string());
                                props
                            },
                        },
                        PtfTreeInstance {
                            position: Vec3::new(80.0, 120.0, 0.0),
                            rotation: -0.785398,
                            scale: 0.7,
                            health_factor: 0.6,
                            tags: vec!["stunted".to_string(), "rocky_soil".to_string()],
                            custom_properties: {
                                let mut props = HashMap::new();
                                props.insert("age_years".to_string(), "25".to_string());
                                props.insert("elevation".to_string(), "25.0".to_string());
                                props.insert("soil_type".to_string(), "rocky".to_string());
                                props
                            },
                        },
                    ],
                },
                PtfTreeGroup {
                    tree_file: "sample_Willow_tree.pt".to_string(),
                    instances: vec![
                        PtfTreeInstance {
                            position: Vec3::new(-120.0, -60.0, 0.0),
                            rotation: 1.570796,
                            scale: 1.1,
                            health_factor: 0.9,
                            tags: vec!["healthy".to_string(), "water_access".to_string()],
                            custom_properties: {
                                let mut props = HashMap::new();
                                props.insert("age_years".to_string(), "35".to_string());
                                props.insert("water_distance".to_string(), "5.0".to_string());
                                props
                            },
                        },
                        PtfTreeInstance {
                            position: Vec3::new(120.0, -60.0, 0.0),
                            rotation: -1.570796,
                            scale: 1.0,
                            health_factor: 1.0,
                            tags: vec!["healthy".to_string(), "specimen".to_string()],
                            custom_properties: {
                                let mut props = HashMap::new();
                                props.insert("age_years".to_string(), "40".to_string());
                                props.insert("water_distance".to_string(), "10.0".to_string());
                                props
                            },
                        },
                        PtfTreeInstance {
                            position: Vec3::new(0.0, 150.0, 0.0),
                            rotation: 0.0,
                            scale: 1.5,
                            health_factor: 1.0,
                            tags: vec!["healthy".to_string(), "crown_jewel".to_string()],
                            custom_properties: {
                                let mut props = HashMap::new();
                                props.insert("age_years".to_string(), "60".to_string());
                                props.insert("water_distance".to_string(), "2.0".to_string());
                                props.insert("showcase".to_string(), "true".to_string());
                                props
                            },
                        },
                    ],
                },
            ],
        }
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

    #[test]
    fn test_stf_parsing() {
        let stf_content = r#"oak_tree_01.srt
3
100.0 50.0 0.0 1.5708 1.2
-50.0 100.0 10.0 0.0 0.8
200.0 -30.0 5.0 -0.7854 1.5"#;

        let forest = StfForestLoader::parse_stf_content(stf_content, "test_forest".to_string()).unwrap();
        
        assert_eq!(forest.name, "test_forest");
        assert_eq!(forest.instances.len(), 3);
        
        let first_instance = &forest.instances[0];
        assert_eq!(first_instance.tree_file, "oak_tree_01.srt");
        assert_eq!(first_instance.position, Vec3::new(100.0, 50.0, 0.0));
        assert!((first_instance.rotation - 1.5708).abs() < 0.001);
        assert!((first_instance.scale - 1.2).abs() < 0.001);
    }

    #[test]
    fn test_stf_parsing_multiple_tree_types() {
        let stf_content = r#"oak_tree.srt
2
0.0 0.0 0.0 0.0 1.0
50.0 0.0 0.0 1.0 1.1
pine_tree.srt
1
100.0 200.0 0.0 -1.5708 0.9"#;

        let forest = StfForestLoader::parse_stf_content(stf_content, "mixed_forest".to_string()).unwrap();
        
        assert_eq!(forest.instances.len(), 3);
        assert_eq!(forest.instances[0].tree_file, "oak_tree.srt");
        assert_eq!(forest.instances[1].tree_file, "oak_tree.srt");
        assert_eq!(forest.instances[2].tree_file, "pine_tree.srt");
        
        // Test positions and scales
        assert_eq!(forest.instances[0].position, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(forest.instances[1].position, Vec3::new(50.0, 0.0, 0.0));
        assert_eq!(forest.instances[2].position, Vec3::new(100.0, 200.0, 0.0));
        
        assert!((forest.instances[1].scale - 1.1).abs() < 0.001);
        assert!((forest.instances[2].scale - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_stf_parsing_invalid_format() {
        // Missing instance count
        let invalid_stf = r#"oak_tree.srt
100.0 50.0 0.0 0.0 1.0"#;

        let result = StfForestLoader::parse_stf_content(invalid_stf, "test".to_string());
        assert!(result.is_err());
        
        // Wrong number of values per instance
        let invalid_stf2 = r#"oak_tree.srt
1
100.0 50.0 0.0"#; // Missing rotation and scale

        let result2 = StfForestLoader::parse_stf_content(invalid_stf2, "test".to_string());
        assert!(result2.is_err());
    }

    #[test]
    fn test_batch_forest_generation_from_stf() {
        let stf_content = r#"test_tree.srt
2
0.0 0.0 0.0 0.0 1.0
100.0 100.0 0.0 1.5708 1.5"#;

        // Create a temporary STF file for testing
        let temp_path = std::env::temp_dir().join("test_forest.stf");
        std::fs::write(&temp_path, stf_content).unwrap();
        
        let batch_generator = BatchTreeGenerator::new(GenerationParams::default());
        let result = batch_generator.generate_forest_from_stf(&temp_path);
        
        assert!(result.is_ok());
        let trees = result.unwrap();
        assert_eq!(trees.len(), 2);
        
        // Verify trees have different characteristics due to different positions and scales
        let (pos1, tree1) = &trees[0];
        let (pos2, tree2) = &trees[1];
        
        assert_eq!(*pos1, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(*pos2, Vec3::new(100.0, 100.0, 0.0));
        
        // Trees should have different heights due to different scales
        assert_ne!(tree1.trunk.height, tree2.trunk.height);
        
        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_srt_basic_parsing() {
        // Test basic SRT file parsing with minimal data
        let dummy_data = vec![0u8; 32]; // Minimal valid file size
        
        let result = SpeedTreeImporter::parse_srt_data(&dummy_data, "test.srt".to_string());
        assert!(result.is_ok());
        
        let imported = result.unwrap();
        assert_eq!(imported.source_file, "test.srt");
        assert!(!imported.trunk_segments.is_empty());
        assert!(!imported.lod_levels.is_empty());
        assert_eq!(imported.lod_levels.len(), 4); // Should have 4 LOD levels
    }

    #[test]
    fn test_srt_too_small() {
        let small_data = vec![0u8; 8]; // Too small
        
        let result = SpeedTreeImporter::parse_srt_data(&small_data, "small.srt".to_string());
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ImportError::InvalidData(_) => {}, // Expected error type
            _ => panic!("Expected InvalidData error"),
        }
    }

    #[test]
    fn test_create_tree_from_imported_data() {
        let imported_data = ImportedTreeData {
            trunk_segments: vec![
                TrunkSegment {
                    start: Vec2::new(0.0, 0.0),
                    end: Vec2::new(0.0, 50.0),
                    width: 10.0,
                },
                TrunkSegment {
                    start: Vec2::new(0.0, 50.0),
                    end: Vec2::new(0.0, 100.0),
                    width: 8.0,
                },
            ],
            branches: vec![
                Branch {
                    start: Vec2::new(0.0, 60.0),
                    end: Vec2::new(30.0, 80.0),
                    thickness: 4.0,
                    generation: 1,
                    parent: None,
                },
                Branch {
                    start: Vec2::new(20.0, 75.0),
                    end: Vec2::new(45.0, 90.0),
                    thickness: 2.0,
                    generation: 2,
                    parent: Some(0),
                },
            ],
            lod_levels: vec![
                LodData { level: 0, max_distance: 50.0, branch_count: 2, leaf_density: 1.0 },
                LodData { level: 1, max_distance: 100.0, branch_count: 1, leaf_density: 0.5 },
            ],
            source_file: "test.srt".to_string(),
        };

        let wind_params = WindParams {
            strength: 2.0,
            frequency: 1.0,
            turbulence: 0.2,
            direction: Vec2::new(1.0, 0.0),
        };

        // Test high LOD
        let tree_lod0 = SpeedTreeImporter::create_tree_from_imported_data(&imported_data, wind_params.clone(), 0);
        assert_eq!(tree_lod0.lod_level, 0);
        assert_eq!(tree_lod0.branches.len(), 2); // Should have all branches
        assert!(!tree_lod0.leaves.leaves.is_empty()); // Should have leaves
        assert_eq!(tree_lod0.trunk.height, 100.0);
        assert_eq!(tree_lod0.trunk.base_width, 10.0);

        // Test medium LOD
        let tree_lod1 = SpeedTreeImporter::create_tree_from_imported_data(&imported_data, wind_params.clone(), 1);
        assert_eq!(tree_lod1.lod_level, 1);
        assert_eq!(tree_lod1.branches.len(), 1); // Should have fewer branches

        // Test minimal LOD
        let tree_lod3 = SpeedTreeImporter::create_tree_from_imported_data(&imported_data, wind_params, 3);
        assert_eq!(tree_lod3.lod_level, 3);
        assert_eq!(tree_lod3.branches.len(), 0); // Should have no branches
        assert!(tree_lod3.leaves.leaves.is_empty()); // Should have no leaves
    }

    #[test]
    fn test_hash_string_deterministic() {
        let hash1 = StfForestLoader::hash_string("oak_tree_01.srt");
        let hash2 = StfForestLoader::hash_string("oak_tree_01.srt");
        let hash3 = StfForestLoader::hash_string("pine_tree_02.srt");
        
        assert_eq!(hash1, hash2); // Same string should produce same hash
        assert_ne!(hash1, hash3); // Different strings should produce different hashes
    }

    #[test]
    fn test_pt_file_save_load_cycle() {
        let params = GenerationParams::default();
        let mut generator = TreeGenerator::new(12345);
        let tree = generator.generate(&params);
        
        let temp_path = std::env::temp_dir().join("test_tree.pt");
        
        // Save the tree
        let save_result = PtFileManager::save_tree(&tree, &params, &temp_path, None);
        assert!(save_result.is_ok(), "Failed to save tree: {:?}", save_result);
        
        // Load the tree back
        let load_result = PtFileManager::load_tree(&temp_path);
        assert!(load_result.is_ok(), "Failed to load tree: {:?}", load_result);
        
        let (loaded_tree, loaded_params) = load_result.unwrap();
        
        // Verify basic properties match
        assert_eq!(tree.trunk.height, loaded_tree.trunk.height);
        assert_eq!(tree.trunk.base_width, loaded_tree.trunk.base_width);
        assert_eq!(tree.branches.len(), loaded_tree.branches.len());
        assert_eq!(tree.leaves.leaves.len(), loaded_tree.leaves.leaves.len());
        assert_eq!(tree.template, loaded_tree.template);
        assert_eq!(params.template, loaded_params.template);
        
        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_pt_file_header_validation() {
        // Create a tree with invalid header
        let invalid_data = b"INVALID_HEADER";
        let temp_path = std::env::temp_dir().join("invalid_tree.pt");
        std::fs::write(&temp_path, invalid_data).unwrap();
        
        let result = PtFileManager::load_tree(&temp_path);
        assert!(result.is_err());
        
        if let Err(PtFormatError::InvalidHeader(_)) = result {
            // Expected error type
        } else if let Err(PtFormatError::SerializationError(_)) = result {
            // Also acceptable - bincode deserialize failure
        } else {
            panic!("Expected InvalidHeader or SerializationError, got: {:?}", result);
        }
        
        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_pt_geometry_conversion() {
        let params = GenerationParams::default();
        let mut generator = TreeGenerator::new(54321);
        let original_tree = generator.generate(&params);
        
        // Convert to PT geometry format
        let geometry_data = PtFileManager::convert_tree_to_geometry(&original_tree);
        
        // Verify conversion preserved key properties
        assert_eq!(geometry_data.trunk_segments.len(), original_tree.trunk.segments.len());
        assert_eq!(geometry_data.branches.len(), original_tree.branches.len());
        assert_eq!(geometry_data.leaves.len(), original_tree.leaves.leaves.len());
        
        // Verify bounding box was calculated
        assert!(geometry_data.bounding_box.height > 0.0);
        assert!(geometry_data.bounding_box.max[0] > geometry_data.bounding_box.min[0]);
        assert!(geometry_data.bounding_box.max[1] > geometry_data.bounding_box.min[1]);
        
        // Convert back to tree
        let reconstructed_tree = PtFileManager::convert_geometry_to_tree(
            geometry_data,
            &original_tree.wind_params,
            &PtFileManager::default_metadata(&original_tree.template)
        );
        
        // Verify reconstruction preserved structure
        assert_eq!(original_tree.trunk.segments.len(), reconstructed_tree.trunk.segments.len());
        assert_eq!(original_tree.branches.len(), reconstructed_tree.branches.len());
        assert_eq!(original_tree.leaves.leaves.len(), reconstructed_tree.leaves.leaves.len());
    }

    #[test]
    fn test_pt_material_extraction() {
        let params = GenerationParams::default();
        let mut generator = TreeGenerator::new(98765);
        let tree = generator.generate(&params);
        
        let materials = PtFileManager::extract_materials(&tree);
        
        // Should have trunk, branch, and leaf materials
        assert!(!materials.trunk_materials.is_empty());
        assert!(!materials.branch_materials.is_empty());
        assert!(!materials.leaf_materials.is_empty());
        
        // Check material properties are sensible
        let trunk_material = &materials.trunk_materials[0];
        assert_eq!(trunk_material.name, "Trunk");
        assert!(trunk_material.roughness > 0.0);
        assert!(trunk_material.roughness <= 1.0);
        assert!(trunk_material.metallic >= 0.0);
        assert!(trunk_material.metallic <= 1.0);
    }

    #[test]
    fn test_pt_lod_levels() {
        let params = GenerationParams::default();
        let mut generator = TreeGenerator::new(13579);
        let tree = generator.generate(&params);
        
        let lod_levels = PtFileManager::create_lod_levels(&tree);
        
        // Should have 4 LOD levels
        assert_eq!(lod_levels.len(), 4);
        
        // LOD levels should be ordered by distance
        for i in 1..lod_levels.len() {
            assert!(lod_levels[i].max_distance >= lod_levels[i-1].max_distance);
        }
        
        // Highest LOD should have all branches
        assert_eq!(lod_levels[0].branch_count, tree.branches.len());
        assert_eq!(lod_levels[0].leaf_density, 1.0);
        
        // Lowest LOD should have no branches/leaves
        assert_eq!(lod_levels[3].branch_count, 0);
        assert_eq!(lod_levels[3].leaf_density, 0.0);
    }

    #[test]
    fn test_pt_metadata_creation() {
        let metadata = PtFileManager::default_metadata(&TreeTemplate::Oak);
        
        assert!(metadata.tree_name.contains("Oak"));
        assert_eq!(metadata.tree_type, TreeTemplate::Oak);
        assert!(!metadata.author.is_empty());
        assert!(!metadata.description.is_empty());
        assert!(!metadata.tags.is_empty());
        assert!(metadata.tags.contains(&"procedural".to_string()));
        assert!(metadata.estimated_age_years.is_some());
        assert!(metadata.biome.is_some());
    }

    #[test] 
    fn test_pt_file_different_templates() {
        let templates = vec![TreeTemplate::Oak, TreeTemplate::Pine, TreeTemplate::Willow];
        let temp_dir = std::env::temp_dir();
        
        for template in templates {
            let params = GenerationParams {
                template,
                seed: 42,
                ..default()
            };
            
            let mut generator = TreeGenerator::new(params.seed);
            let tree = generator.generate(&params);
            
            let filename = temp_dir.join(format!("test_{:?}.pt", template));
            
            // Save and load
            assert!(PtFileManager::save_tree(&tree, &params, &filename, None).is_ok());
            let (loaded_tree, loaded_params) = PtFileManager::load_tree(&filename).unwrap();
            
            // Verify template was preserved
            assert_eq!(tree.template, loaded_tree.template);
            assert_eq!(params.template, loaded_params.template);
            
            // Clean up
            std::fs::remove_file(filename).ok();
        }
    }

    #[test]
    #[ignore] // This test creates sample files - run with: cargo test create_sample_files -- --ignored
    fn create_sample_tree_files() {
        use chrono::Utc;
        
        let templates = vec![
            (TreeTemplate::Oak, "Majestic Oak"),
            (TreeTemplate::Pine, "Tall Pine"),
            (TreeTemplate::Willow, "Weeping Willow"),
            (TreeTemplate::Default, "Standard Tree"),
        ];
        
        for (template, name) in templates {
            let params = GenerationParams {
                template,
                seed: match template {
                    TreeTemplate::Oak => 12345,
                    TreeTemplate::Pine => 54321,
                    TreeTemplate::Willow => 98765,
                    TreeTemplate::Default => 13579,
                    _ => 42,
                },
                height_range: match template {
                    TreeTemplate::Pine => (120.0, 180.0),
                    TreeTemplate::Oak => (80.0, 120.0),
                    TreeTemplate::Willow => (100.0, 140.0),
                    _ => (90.0, 130.0),
                },
                branch_angle_variance: match template {
                    TreeTemplate::Pine => 20.0,
                    TreeTemplate::Willow => 60.0,
                    _ => 45.0,
                },
                wind_params: WindParams {
                    strength: 2.0,
                    frequency: 1.2,
                    turbulence: 0.4,
                    direction: Vec2::new(1.0, 0.2),
                },
                leaf_density: match template {
                    TreeTemplate::Pine => 0.8,
                    TreeTemplate::Willow => 1.2,
                    _ => 1.0,
                },
                ..default()
            };
            
            let mut generator = TreeGenerator::new(params.seed);
            let tree = generator.generate(&params);
            
            // Create detailed metadata
            let metadata = PtMetadata {
                tree_name: name.to_string(),
                author: "PixelTree Demo".to_string(),
                created_date: Utc::now(),
                modified_date: Utc::now(),
                description: format!("Sample {} tree for PixelTree demo", name),
                tags: vec!["sample".to_string(), "demo".to_string(), "pixeltree".to_string()],
                tree_type: template,
                estimated_age_years: Some(match template {
                    TreeTemplate::Oak => 45.0,
                    TreeTemplate::Pine => 35.0,
                    TreeTemplate::Willow => 25.0,
                    _ => 30.0,
                }),
                biome: Some("Temperate Forest".to_string()),
                custom_properties: {
                    let mut props = HashMap::new();
                    props.insert("demo_version".to_string(), "1.0".to_string());
                    props.insert("complexity".to_string(), 
                        format!("{} branches, {} leaves", tree.branches.len(), tree.leaves.leaves.len()));
                    props
                },
            };
            
            let filename = format!("examples/sample_{:?}_tree.pt", template);
            
            match PtFileManager::save_tree(&tree, &params, &filename, Some(metadata)) {
                Ok(()) => {
                    println!("Created sample tree file: {}", filename);
                    
                    // Verify we can load it back
                    match PtFileManager::load_tree(&filename) {
                        Ok((loaded_tree, _)) => {
                            println!("  ✓ Verified: {} branches, {} leaves", 
                                loaded_tree.branches.len(), 
                                loaded_tree.leaves.leaves.len());
                        },
                        Err(e) => println!("  ✗ Failed to load: {}", e),
                    }
                },
                Err(e) => println!("Failed to create {}: {}", filename, e),
            }
        }
    }

    #[test]
    fn test_ptf_text_parsing() {
        let ptf_content = r#"# PixelTree Forest v1.0
# Forest: Test Grove
# Author: Test Author
# Biome: Test Forest
# Season: Autumn

sample_Oak_tree.pt
2
-100.0 0.0 0.0 0.0 1.2 1.0 [healthy,mature] {age_years:45,soil_quality:0.8}
50.0 0.0 0.0 0.523599 0.9 0.8 [healthy] {age_years:30}

sample_Pine_tree.pt
1
0.0 150.0 0.0 0.0 1.5 1.0 [ancient] {age_years:100}
"#;

        let ptf_forest = PtfForestManager::parse_text_format(ptf_content).unwrap();
        
        assert_eq!(ptf_forest.forest_metadata.forest_name, "Test Grove");
        assert_eq!(ptf_forest.forest_metadata.author, "Test Author");
        assert_eq!(ptf_forest.forest_metadata.biome, "Test Forest");
        assert!(matches!(ptf_forest.forest_metadata.season, PtfSeason::Autumn));
        assert_eq!(ptf_forest.forest_metadata.total_trees, 3);
        
        assert_eq!(ptf_forest.tree_groups.len(), 2);
        
        let oak_group = &ptf_forest.tree_groups[0];
        assert_eq!(oak_group.tree_file, "sample_Oak_tree.pt");
        assert_eq!(oak_group.instances.len(), 2);
        
        let first_oak = &oak_group.instances[0];
        assert_eq!(first_oak.position, Vec3::new(-100.0, 0.0, 0.0));
        assert_eq!(first_oak.scale, 1.2);
        assert_eq!(first_oak.health_factor, 1.0);
        assert!(first_oak.tags.contains(&"healthy".to_string()));
        assert!(first_oak.tags.contains(&"mature".to_string()));
        assert_eq!(first_oak.custom_properties.get("age_years"), Some(&"45".to_string()));
    }

    #[test]
    fn test_ptf_instance_parsing() {
        // Test basic instance line
        let instance = PtfForestManager::parse_instance_line(
            "100.0 50.0 0.0 1.570796 1.5 0.9"
        ).unwrap();
        
        assert_eq!(instance.position, Vec3::new(100.0, 50.0, 0.0));
        assert!((instance.rotation - 1.570796).abs() < 0.001);
        assert_eq!(instance.scale, 1.5);
        assert_eq!(instance.health_factor, 0.9);
        assert!(instance.tags.is_empty());
        assert!(instance.custom_properties.is_empty());

        // Test instance with tags
        let instance_with_tags = PtfForestManager::parse_instance_line(
            "0.0 0.0 0.0 0.0 1.0 1.0 [healthy,ancient,landmark]"
        ).unwrap();
        
        assert_eq!(instance_with_tags.tags.len(), 3);
        assert!(instance_with_tags.tags.contains(&"healthy".to_string()));
        assert!(instance_with_tags.tags.contains(&"ancient".to_string()));
        assert!(instance_with_tags.tags.contains(&"landmark".to_string()));

        // Test instance with properties
        let instance_with_props = PtfForestManager::parse_instance_line(
            "0.0 0.0 0.0 0.0 1.0 1.0 {age_years:50,soil_type:clay,water_access:true}"
        ).unwrap();
        
        assert_eq!(instance_with_props.custom_properties.len(), 3);
        assert_eq!(instance_with_props.custom_properties.get("age_years"), Some(&"50".to_string()));
        assert_eq!(instance_with_props.custom_properties.get("soil_type"), Some(&"clay".to_string()));
        assert_eq!(instance_with_props.custom_properties.get("water_access"), Some(&"true".to_string()));

        // Test instance with both tags and properties
        let full_instance = PtfForestManager::parse_instance_line(
            "10.0 20.0 5.0 0.785398 1.2 0.8 [diseased,recovering] {treatment_date:2024-01-15,progress:0.3}"
        ).unwrap();
        
        assert_eq!(full_instance.position, Vec3::new(10.0, 20.0, 5.0));
        assert_eq!(full_instance.tags.len(), 2);
        assert_eq!(full_instance.custom_properties.len(), 2);
    }

    #[test]
    fn test_ptf_save_load_cycle() {
        let sample_ptf = PtfForestManager::create_sample_ptf();
        let temp_path = std::env::temp_dir().join("test_forest.ptf");
        
        // Test text format
        let save_result = PtfForestManager::save_forest(
            &sample_ptf.forest_metadata,
            &sample_ptf.tree_groups.iter()
                .map(|g| (g.tree_file.clone(), g.instances.clone()))
                .collect::<Vec<_>>(),
            &temp_path,
            PtfEncoding::Text,
        );
        assert!(save_result.is_ok());
        
        let loaded_ptf = PtfForestManager::load_forest(&temp_path).unwrap();
        assert_eq!(loaded_ptf.forest_metadata.forest_name, sample_ptf.forest_metadata.forest_name);
        assert_eq!(loaded_ptf.tree_groups.len(), sample_ptf.tree_groups.len());
        
        // Test binary format
        let binary_path = std::env::temp_dir().join("test_forest_binary.ptf");
        let binary_save = PtfForestManager::save_forest(
            &sample_ptf.forest_metadata,
            &sample_ptf.tree_groups.iter()
                .map(|g| (g.tree_file.clone(), g.instances.clone()))
                .collect::<Vec<_>>(),
            &binary_path,
            PtfEncoding::Binary,
        );
        assert!(binary_save.is_ok());
        
        let loaded_binary = PtfForestManager::load_forest(&binary_path).unwrap();
        assert_eq!(loaded_binary.forest_metadata.forest_name, sample_ptf.forest_metadata.forest_name);
        
        // Clean up
        std::fs::remove_file(temp_path).ok();
        std::fs::remove_file(binary_path).ok();
    }

    #[test]
    fn test_ptf_seasonal_modifications() {
        let mut tree = PixelTree {
            trunk: TrunkData {
                base_pos: Vec2::ZERO,
                height: 100.0,
                base_width: 8.0,
                segments: vec![],
            },
            branches: vec![],
            leaves: LeafCluster {
                leaves: vec![
                    Leaf {
                        pos: Vec2::new(0.0, 50.0),
                        rot: 0.0,
                        scale: 1.0,
                        leaf_type: 0,
                        color: [34, 139, 34], // Forest green
                    },
                    Leaf {
                        pos: Vec2::new(10.0, 60.0),
                        rot: 0.0,
                        scale: 1.0,
                        leaf_type: 1,
                        color: [34, 139, 34],
                    },
                    Leaf {
                        pos: Vec2::new(-10.0, 65.0),
                        rot: 0.0,
                        scale: 1.0,
                        leaf_type: 2,
                        color: [34, 139, 34],
                    },
                ]
            },
            wind_params: WindParams::default(),
            lod_level: 0,
            template: TreeTemplate::Default,
        };

        let instance = PtfTreeInstance {
            position: Vec3::ZERO,
            rotation: 0.0,
            scale: 1.0,
            health_factor: 1.0,
            tags: vec![],
            custom_properties: HashMap::new(),
        };

        // Test autumn modifications
        let autumn_metadata = PtfForestMetadata {
            season: PtfSeason::Autumn,
            global_wind: WindParams::default(),
            forest_name: "Test".to_string(),
            author: "Test".to_string(),
            created_date: Utc::now(),
            modified_date: Utc::now(),
            biome: "Test".to_string(),
            time_of_day: None,
            terrain: None,
            lighting: None,
            weather: None,
            total_trees: 1,
            bounding_box: PtBoundingBox { min: [0.0, 0.0], max: [100.0, 100.0], height: 100.0 },
            tags: vec![],
            custom_properties: HashMap::new(),
        };

        let original_leaf_count = tree.leaves.leaves.len();
        PtfForestManager::apply_instance_modifications(&mut tree, &instance, &autumn_metadata);
        
        // Should still have same number of leaves but colors changed
        assert_eq!(tree.leaves.leaves.len(), original_leaf_count);
        // Check that autumn colors were applied
        assert!(tree.leaves.leaves.iter().any(|leaf| 
            leaf.color == [255, 69, 0] || leaf.color == [255, 215, 0] || leaf.color == [205, 92, 92]
        ));

        // Test winter modifications
        let mut winter_tree = tree.clone();
        let winter_metadata = PtfForestMetadata {
            season: PtfSeason::Winter,
            ..autumn_metadata.clone()
        };

        PtfForestManager::apply_instance_modifications(&mut winter_tree, &instance, &winter_metadata);
        
        // Should have much fewer leaves in winter
        assert!(winter_tree.leaves.leaves.len() < original_leaf_count);
        assert!(winter_tree.leaves.leaves.len() <= original_leaf_count / 10);
    }

    #[test]
    fn test_ptf_health_factor_modifications() {
        let mut healthy_tree = generate_test_tree();
        let mut diseased_tree = healthy_tree.clone();
        
        let healthy_instance = PtfTreeInstance {
            position: Vec3::ZERO,
            rotation: 0.0,
            scale: 1.0,
            health_factor: 1.0,
            tags: vec!["healthy".to_string()],
            custom_properties: HashMap::new(),
        };
        
        let diseased_instance = PtfTreeInstance {
            position: Vec3::ZERO,
            rotation: 0.0,
            scale: 1.0,
            health_factor: 0.4, // Unhealthy
            tags: vec!["diseased".to_string()],
            custom_properties: HashMap::new(),
        };

        let metadata = create_test_metadata();
        
        let original_leaf_count = healthy_tree.leaves.leaves.len();
        
        // Apply modifications
        PtfForestManager::apply_instance_modifications(&mut healthy_tree, &healthy_instance, &metadata);
        PtfForestManager::apply_instance_modifications(&mut diseased_tree, &diseased_instance, &metadata);
        
        // Diseased tree should have fewer leaves
        assert!(diseased_tree.leaves.leaves.len() < healthy_tree.leaves.leaves.len());
        assert!(diseased_tree.leaves.leaves.len() < (original_leaf_count as f32 * 0.5) as usize);
        
        // Diseased tree should have different leaf colors (more brown/yellow)
        let diseased_colors = diseased_tree.leaves.leaves.iter()
            .map(|leaf| leaf.color)
            .collect::<Vec<_>>();
        let healthy_colors = healthy_tree.leaves.leaves.iter()
            .map(|leaf| leaf.color)
            .collect::<Vec<_>>();
        
        // At least some colors should be different
        assert_ne!(diseased_colors, healthy_colors);
    }

    fn generate_test_tree() -> PixelTree {
        let mut generator = TreeGenerator::new(12345);
        generator.generate(&GenerationParams::default())
    }

    fn create_test_metadata() -> PtfForestMetadata {
        PtfForestMetadata {
            forest_name: "Test Forest".to_string(),
            author: "Test".to_string(),
            created_date: Utc::now(),
            modified_date: Utc::now(),
            biome: "Test Biome".to_string(),
            season: PtfSeason::Summer,
            time_of_day: None,
            global_wind: WindParams::default(),
            terrain: None,
            lighting: None,
            weather: None,
            total_trees: 1,
            bounding_box: PtBoundingBox { min: [0.0, 0.0], max: [100.0, 100.0], height: 100.0 },
            tags: vec![],
            custom_properties: HashMap::new(),
        }
    }

    #[test]
    #[ignore] // Create sample PTF files
    fn create_sample_ptf_files() {
        let sample_ptf = PtfForestManager::create_sample_ptf();
        
        // Create text format
        match PtfForestManager::save_forest(
            &sample_ptf.forest_metadata,
            &sample_ptf.tree_groups.iter()
                .map(|g| (g.tree_file.clone(), g.instances.clone()))
                .collect::<Vec<_>>(),
            "examples/ancient_grove.ptf",
            PtfEncoding::Text,
        ) {
            Ok(()) => println!("Created sample PTF file: examples/ancient_grove.ptf"),
            Err(e) => println!("Failed to create PTF file: {}", e),
        }

        // Create a winter variant
        let mut winter_ptf = sample_ptf.clone();
        winter_ptf.forest_metadata.season = PtfSeason::Winter;
        winter_ptf.forest_metadata.forest_name = "Winter Grove".to_string();
        
        match PtfForestManager::save_forest(
            &winter_ptf.forest_metadata,
            &winter_ptf.tree_groups.iter()
                .map(|g| (g.tree_file.clone(), g.instances.clone()))
                .collect::<Vec<_>>(),
            "examples/winter_grove.ptf",
            PtfEncoding::Text,
        ) {
            Ok(()) => println!("Created winter PTF file: examples/winter_grove.ptf"),
            Err(e) => println!("Failed to create winter PTF file: {}", e),
        }

        // Create a JSON format for tooling
        match PtfForestManager::save_forest(
            &sample_ptf.forest_metadata,
            &sample_ptf.tree_groups.iter()
                .map(|g| (g.tree_file.clone(), g.instances.clone()))
                .collect::<Vec<_>>(),
            "examples/ancient_grove.json",
            PtfEncoding::Json,
        ) {
            Ok(()) => println!("Created JSON PTF file: examples/ancient_grove.json"),
            Err(e) => println!("Failed to create JSON PTF file: {}", e),
        }
    }

    // ============================================================================
    // ERROR HANDLING AND SECURITY TESTS
    // ============================================================================

    #[test]
    fn test_pt_path_traversal_protection() {
        let tree = generate_test_tree();
        let params = GenerationParams::default();
        
        // Test various path traversal attempts
        let malicious_paths = vec![
            "../../../etc/passwd",
            "..\\windows\\system32\\config",
            "/etc/shadow",
            "test/../../../sensitive.txt",
            "normal_path\0hidden_path",
            &"a".repeat(2000), // Very long path
        ];
        
        for path in malicious_paths {
            let result = PtFileManager::save_tree(&tree, &params, path);
            assert!(result.is_err(), "Should reject malicious path: {}", path);
        }
    }

    #[test]
    fn test_ptf_path_traversal_protection() {
        let metadata = create_test_metadata();
        let tree_groups = vec![("test.pt".to_string(), vec![])];
        
        let malicious_paths = vec![
            "../../../etc/passwd.ptf",
            "..\\windows\\system32\\config.ptf",
            "/etc/shadow.ptf",
            "test/../../../sensitive.ptf",
            "normal_path\0hidden_path.ptf",
            &format!("{}.ptf", "a".repeat(2000)),
        ];
        
        for path in malicious_paths {
            let result = PtfForestManager::save_forest(
                &metadata, 
                &tree_groups, 
                path, 
                PtfEncoding::Json
            );
            assert!(result.is_err(), "Should reject malicious PTF path: {}", path);
        }
    }

    #[test]
    fn test_pt_oversized_data_rejection() {
        // Create tree with excessive branches
        let mut oversized_tree = generate_test_tree();
        
        // Add way too many branches
        for i in 0..15000 {  // Exceeds MAX_BRANCHES
            oversized_tree.trunk.segments.push(TrunkSegment {
                start_pos: Vec2::new(i as f32, 0.0),
                end_pos: Vec2::new(i as f32 + 1.0, 1.0),
                width: 1.0,
                color: [139, 69, 19],
            });
        }
        
        let params = GenerationParams::default();
        let result = PtFileManager::save_tree(&oversized_tree, &params, "test_oversized.pt");
        assert!(result.is_err(), "Should reject oversized tree data");
    }

    #[test]
    fn test_ptf_oversized_data_rejection() {
        let mut metadata = create_test_metadata();
        
        // Add excessive custom properties
        for i in 0..15000 {  // Exceeds MAX_PTF_CUSTOM_PROPERTIES
            metadata.custom_properties.insert(
                format!("prop_{}", i),
                format!("value_{}", i)
            );
        }
        
        let tree_groups = vec![("test.pt".to_string(), vec![])];
        let result = PtfForestManager::save_forest(
            &metadata, 
            &tree_groups, 
            "test_oversized.ptf", 
            PtfEncoding::Json
        );
        assert!(result.is_err(), "Should reject oversized PTF metadata");
    }

    #[test]
    fn test_pt_invalid_string_lengths() {
        let tree = generate_test_tree();
        let mut params = GenerationParams::default();
        
        // Test oversized strings in metadata  
        params.metadata.tree_name = "a".repeat(2000);  // Exceeds MAX_TREE_NAME_LEN
        let result = PtFileManager::save_tree(&tree, &params, "test_long_name.pt");
        assert!(result.is_err(), "Should reject oversized tree name");
        
        params.metadata.tree_name = "Normal".to_string();
        params.metadata.description = "a".repeat(2000);  // Exceeds MAX_DESCRIPTION_LEN
        let result = PtFileManager::save_tree(&tree, &params, "test_long_desc.pt");
        assert!(result.is_err(), "Should reject oversized description");
    }

    #[test]
    fn test_ptf_invalid_tree_instances() {
        let metadata = create_test_metadata();
        
        // Test invalid scale
        let invalid_instance = PtfTreeInstance {
            position: Vec3::ZERO,
            rotation: 0.0,
            scale: -1.0,  // Invalid negative scale
            health_factor: 1.0,
            tags: vec![],
            custom_properties: HashMap::new(),
        };
        
        let tree_groups = vec![("test.pt".to_string(), vec![invalid_instance])];
        let result = PtfForestManager::save_forest(
            &metadata, 
            &tree_groups, 
            "test_invalid.ptf", 
            PtfEncoding::Json
        );
        assert!(result.is_err(), "Should reject invalid tree instance scale");
        
        // Test invalid health factor
        let invalid_health = PtfTreeInstance {
            position: Vec3::ZERO,
            rotation: 0.0,
            scale: 1.0,
            health_factor: 2.0,  // Invalid > 1.0
            tags: vec![],
            custom_properties: HashMap::new(),
        };
        
        let tree_groups = vec![("test.pt".to_string(), vec![invalid_health])];
        let result = PtfForestManager::save_forest(
            &metadata, 
            &tree_groups, 
            "test_invalid_health.ptf", 
            PtfEncoding::Json
        );
        assert!(result.is_err(), "Should reject invalid health factor");
    }

    #[test] 
    fn test_pt_file_corruption_detection() {
        use std::io::Write;
        
        let tree = generate_test_tree();
        let params = GenerationParams::default();
        let temp_path = "test_corrupt.pt";
        
        // First save a valid file
        PtFileManager::save_tree(&tree, &params, temp_path).unwrap();
        
        // Corrupt the file by truncating it
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(temp_path)
            .unwrap();
        file.write_all(b"corrupted").unwrap();
        drop(file);
        
        // Try to load corrupted file
        let result = PtFileManager::load_tree(temp_path);
        assert!(result.is_err(), "Should detect corrupted PT file");
        
        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_ptf_invalid_bounding_box() {
        let mut metadata = create_test_metadata();
        
        // Test invalid bounding box (min > max)
        metadata.bounding_box = PtBoundingBox {
            min: [100.0, 100.0],
            max: [0.0, 0.0],  // Invalid: max < min
            height: 100.0,
        };
        
        let tree_groups = vec![("test.pt".to_string(), vec![])];
        let result = PtfForestManager::save_forest(
            &metadata, 
            &tree_groups, 
            "test_invalid_bbox.ptf", 
            PtfEncoding::Json
        );
        assert!(result.is_err(), "Should reject invalid bounding box");
        
        // Test invalid height
        metadata.bounding_box = PtBoundingBox {
            min: [0.0, 0.0],
            max: [100.0, 100.0],
            height: -1.0,  // Invalid negative height
        };
        
        let result = PtfForestManager::save_forest(
            &metadata, 
            &tree_groups, 
            "test_invalid_height.ptf", 
            PtfEncoding::Json
        );
        assert!(result.is_err(), "Should reject invalid bounding box height");
    }

    #[test]
    fn test_atomic_file_operations() {
        let tree = generate_test_tree();
        let params = GenerationParams::default();
        let test_path = "test_atomic.pt";
        
        // Save tree
        PtFileManager::save_tree(&tree, &params, test_path).unwrap();
        
        // Verify no temporary files left behind
        assert!(!std::path::Path::new("test_atomic.pt.tmp").exists());
        
        // Verify we can load the file
        let (loaded_tree, _) = PtFileManager::load_tree(test_path).unwrap();
        assert_eq!(loaded_tree.trunk.segments.len(), tree.trunk.segments.len());
        
        // Cleanup
        let _ = std::fs::remove_file(test_path);
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