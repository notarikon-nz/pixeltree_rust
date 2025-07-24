# PixelTree 🌳

A high-performance procedural 2D tree generator for pixel-art games, built with Bevy. Generate beautiful, wind-animated trees with automatic LOD and spatial culling.

![Rust](https://img.shields.io/badge/rust-1.75%2B-orange)
![Bevy](https://img.shields.io/badge/bevy-0.16.1-green)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Features

### Performance First
- **Zero-cost abstractions** with minimal dependencies
- **Parallel generation** using Rayon for forests
- **Spatial indexing** for efficient culling of off-screen trees
- **Pre-computed wind tables** eliminate per-frame trigonometry
- **Memory-optimized** leaf data structures with ~40% less memory usage

### Procedural Generation
- **Deterministic generation** - same position always yields same tree
- **Customizable parameters** - height, width, branching, leaf density
- **Multiple tree templates** - Default, Pine, Oak, Willow
- **Stack-based branch generation** - no recursion overhead

### Game-Ready
- **Automatic LOD system** with 4 detail levels
- **Wind animation** with customizable strength and turbulence  
- **GPU instancing support** for rendering thousands of trees
- **Compact serialization** for save files and networking

## Quick Start

```rust
use bevy::prelude::*;
use pixeltree::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, PixelTreePlugin))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut spatial_index: ResMut<TreeSpatialIndex>,
) {
    // Spawn a single tree
    let params = GenerationParams {
        height_range: (80.0, 120.0),
        wind_params: WindParams {
            strength: 2.0,
            frequency: 1.5,
            turbulence: 0.3,
            direction: Vec2::new(1.0, 0.0),
        },
        ..default()
    };
    
    spawn_tree(
        &mut commands, 
        Vec3::new(0.0, 0.0, 0.0), 
        params,
        &mut spatial_index,
    );
}
```

## Advanced Usage

### Generate a Forest

```rust
// Generate 100 trees in parallel
let positions: Vec<Vec2> = (0..100)
    .map(|i| Vec2::new(
        (i % 10) as f32 * 50.0 - 250.0,
        (i / 10) as f32 * 50.0 - 250.0,
    ))
    .collect();

spawn_forest(&mut commands, positions, params, &mut spatial_index);
```

### Deterministic Generation

```rust
// Trees at the same position will always look identical
let tree = generate_tree_deterministic(
    Vec2::new(100.0, 50.0), 
    &params
);
```

### Custom Tree Templates

```rust
let params = GenerationParams {
    template: TreeTemplate::Pine,
    height_range: (100.0, 150.0),
    branch_angle_variance: 25.0, // More upright branches
    ..default()
};
```

### Query Visible Trees

```rust
fn render_visible_trees(
    spatial_index: Res<TreeSpatialIndex>,
    camera: Query<&Transform, With<Camera>>,
) {
    let camera_pos = camera.single().translation.truncate();
    let view_bounds = Rect::from_center_size(camera_pos, Vec2::splat(500.0));
    
    let visible_trees = spatial_index.get_visible_trees(view_bounds);
    // Only process visible trees...
}
```

## Cargo Dependencies

```toml
[dependencies]
bevy = "0.16.1"
rand = "0.8"
serde = { version = "1.0", features = ["derive"] }
rayon = "1.7"
```

## License

MIT License - see [LICENSE](LICENSE) for details.