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

#[derive(Component, Clone, Debug, Reflect, Serialize, Deserialize)]
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
}

// ============================================================================
// PT FORMAT IMPLEMENTATION
// ============================================================================

pub struct PtFileManager;

impl PtFileManager {
    pub fn save_tree<P: AsRef<Path>>(
        tree: &PixelTree,
        params: &GenerationParams,
        path: P,
        metadata: Option<PtMetadata>,
    ) -> Result<(), PtFormatError> {
        let geometry_data = Self::convert_tree_to_geometry(tree);
        let material_data = Self::extract_materials(tree);
        let lod_levels = Self::create_lod_levels(tree);
        
        let metadata = metadata.unwrap_or_else(|| Self::default_metadata(&params.template));
        
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
            metadata,
        };
        
        let serialized = if pt_file.header.compression {
            // Use bincode with compression for efficiency
            bincode::serialize(&pt_file)?
        } else {
            // For debugging, use JSON
            serde_json::to_vec_pretty(&pt_file).map_err(|e| {
                PtFormatError::SerializationError(bincode::Error::new(bincode::ErrorKind::Custom(e.to_string())))
            })?
        };
        
        std::fs::write(path, serialized)?;
        Ok(())
    }
    
    pub fn load_tree<P: AsRef<Path>>(path: P) -> Result<(PixelTree, GenerationParams), PtFormatError> {
        let data = std::fs::read(path.as_ref())?;
        
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
        if pt_file.header.magic != "PIXELTREE" {
            return Err(PtFormatError::InvalidHeader(format!(
                "Expected 'PIXELTREE', got '{}'", pt_file.header.magic
            )));
        }
        
        // Version compatibility check
        if pt_file.header.format_version > 1 {
            return Err(PtFormatError::UnsupportedVersion(format!(
                "Format version {} not supported", pt_file.header.format_version
            )));
        }
        
        let tree = if let Some(geometry) = pt_file.geometry_data {
            Self::convert_geometry_to_tree(geometry, &pt_file.wind_data, &pt_file.metadata)
        } else {
            // Regenerate from procedural parameters
            let mut generator = TreeGenerator::new(pt_file.procedural_params.seed);
            generator.generate(&pt_file.procedural_params)
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
    
    fn extract_materials(tree: &PixelTree) -> PtMaterialData {
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