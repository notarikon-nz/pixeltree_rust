use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy_light_2d::prelude::*;
use pixeltree::*;

fn main() {
    println!("🌳 PixelTree Demo Starting...");
    
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "PixelTree Demo".into(),
                    resolution: (1200.0, 800.0).into(),
                    ..default()
                }),
                ..default()
            }),
            PixelTreePlugin,
            Light2dPlugin,
        ))
        .init_resource::<SelectedTree>()
        .init_resource::<ShadowSettings>()
        .init_resource::<LeafSpritePool>()
        .insert_resource(FpsCounter { fps: 0.0, frame_count: 0, timer: 0.0 })
        .add_systems(Startup, setup_demo)
        .add_systems(Update, (
            camera_controls,
            tree_info_display,
            render_trees,
            animate_leaves_and_branches,
        ))
        .add_systems(Update, (
            tree_selection_system,
            handle_regenerated_trees,
            shadow_system,
            fps_system,
            tree_optimization_system,
        ))
        .add_systems(Last, force_high_lod) // Override LOD at the very end of each frame
        .add_systems(Startup, (
            setup_leaf_sprite_pool.after(setup_demo),
            setup_branch_sprites.after(setup_demo),
        ))
        .run();
}

fn setup_demo(
    mut commands: Commands,
    mut spatial_index: ResMut<TreeSpatialIndex>,
) {
    println!("Setting up demo scene...");
    
    // Spawn camera with proper Z positioning
    commands.spawn((
        Camera2d::default(),
        Transform::from_xyz(0.0, 0.0, 1000.0),
        CameraZoom(1.0),
        Light2d::default(),
    ));

    // Add global sunlight
    commands.spawn((
        PointLight2d {
            intensity: 3.0,
            radius: 2000.0,
            ..default()
        },
        Transform::from_xyz(500.0, 800.0, 100.0), // Sun position
    ));

    // Create different tree types across the scene
    let tree_configs = vec![
        (Vec2::new(-300.0, 0.0), TreeTemplate::Oak, "Oak Tree"),
        (Vec2::new(-100.0, 0.0), TreeTemplate::Default, "Default Tree"),
        (Vec2::new(100.0, 0.0), TreeTemplate::Pine, "Pine Tree"),
        (Vec2::new(300.0, 0.0), TreeTemplate::Willow, "Willow Tree"),
    ];
    
    for (pos, template, _name) in tree_configs {
        let params = GenerationParams {
            template,
            height_range: match template {
                TreeTemplate::Pine => (120.0, 180.0),
                TreeTemplate::Oak => (80.0, 120.0),
                TreeTemplate::Willow => (100.0, 140.0),
                _ => (90.0, 130.0),
            },
            wind_params: WindParams {
                strength: 2.5,
                frequency: 1.2,
                turbulence: 0.4,
                direction: Vec2::new(1.0, 0.2),
            },
            branch_angle_variance: match template {
                TreeTemplate::Pine => 20.0,
                TreeTemplate::Willow => 60.0,
                _ => 45.0,
            },
            leaf_density: 3.0, // 3x more leaves
            lod_level: 0, // Ensure high detail
            ..default()
        };
        
        spawn_tree(&mut commands, pos.extend(0.0), params, &mut spatial_index);
        println!("Spawned {} at {:?}", _name, pos);
    }
    
    // Create a small forest in the background
    let forest_positions: Vec<Vec2> = (0..15)
        .map(|i| {
            let angle = i as f32 * std::f32::consts::TAU / 15.0;
            let radius = 500.0 + (i as f32 * 20.0);
            Vec2::new(
                angle.cos() * radius,
                angle.sin() * radius,
            )
        })
        .collect();
    
    let forest_params = GenerationParams {
        height_range: (60.0, 100.0),
        wind_params: WindParams {
            strength: 1.5,
            frequency: 1.0,
            turbulence: 0.2,
            direction: Vec2::new(0.8, 0.3),
        },
        leaf_density: 3.0, // 3x more leaves for forest too
        lod_level: 0, // High detail for all trees initially
        ..default()
    };
    
    spawn_forest(&mut commands, forest_positions, forest_params, &mut spatial_index);
    println!("Spawned background forest with {} trees", 15);
    
    println!("Demo setup complete! Use WASD to move camera.");
}

fn camera_controls(
    mut camera_query: Query<(&mut Transform, &mut CameraZoom), With<Camera>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<MouseWheel>,
    time: Res<Time>,
) {
    if let Ok((mut transform, mut zoom)) = camera_query.single_mut() {
        let speed = 300.0 * time.delta_secs() * zoom.0;
        
        // WASD movement
        if keyboard.pressed(KeyCode::KeyW) || keyboard.pressed(KeyCode::ArrowUp) {
            transform.translation.y += speed;
        }
        if keyboard.pressed(KeyCode::KeyS) || keyboard.pressed(KeyCode::ArrowDown) {
            transform.translation.y -= speed;
        }
        if keyboard.pressed(KeyCode::KeyA) || keyboard.pressed(KeyCode::ArrowLeft) {
            transform.translation.x -= speed;
        }
        if keyboard.pressed(KeyCode::KeyD) || keyboard.pressed(KeyCode::ArrowRight) {
            transform.translation.x += speed;
        }
        
        // Zoom controls via transform scale - Z = zoom in, X = zoom out  
        if keyboard.pressed(KeyCode::KeyZ) {
            zoom.0 = (zoom.0 + time.delta_secs() * 2.0).min(5.0); // Zoom in = larger scale
            transform.scale = Vec3::splat(zoom.0);
        }
        if keyboard.pressed(KeyCode::KeyX) {
            zoom.0 = (zoom.0 - time.delta_secs() * 2.0).max(0.1); // Zoom out = smaller scale
            transform.scale = Vec3::splat(zoom.0);
        }
        
        // Mouse drag controls
        if mouse.pressed(MouseButton::Left) {
            for motion in mouse_motion.read() {
                transform.translation.x -= motion.delta.x * (2.0 / zoom.0); // Adjust sensitivity based on zoom
                transform.translation.y += motion.delta.y * (2.0 / zoom.0);
            }
        }
        
        // Mouse wheel zoom - scroll up = zoom in, scroll down = zoom out
        for wheel in mouse_wheel.read() {
            zoom.0 = (zoom.0 + wheel.y * 0.1).clamp(0.1, 5.0); // Zoom in = larger scale
            transform.scale = Vec3::splat(zoom.0);
        }
        
        // Camera tilt controls (R/F keys for Y-axis rotation)
        if keyboard.pressed(KeyCode::KeyR) {
            let rotation_speed = 1.0 * time.delta_secs();
            transform.rotation = transform.rotation * Quat::from_rotation_z(rotation_speed);
        }
        if keyboard.pressed(KeyCode::KeyF) {
            let rotation_speed = 1.0 * time.delta_secs();
            transform.rotation = transform.rotation * Quat::from_rotation_z(-rotation_speed);
        }
    }
}


#[derive(Component)]
struct TreeInfoText;

#[derive(Component)]
struct CameraZoom(f32);


#[derive(Component)]
struct LeafSprite {
    tree_entity: Entity,
    base_position: Vec2,
    base_rotation: f32,
}

#[derive(Component)]
struct BranchSprite;

#[derive(Component)]
struct ShadowCaster;

#[derive(Component)]
struct RegeneratedTreeMarker {
    entity: Entity,
}

#[derive(Component)]
struct OptimizeTreeMarker {
    entity: Entity,
}

#[derive(Resource, Default)]
struct SelectedTree {
    entity: Option<Entity>,
    distance: f32,
}

#[derive(Resource, Default)]
struct ShadowSettings {
    enabled: bool,
}

#[derive(Resource)]
struct FpsCounter {
    fps: f32,
    frame_count: u32,
    timer: f32,
}

#[derive(Resource)]
struct LeafSpritePool {
    sprites: Vec<Entity>,
    available: Vec<usize>,
    active: Vec<usize>,
}

impl Default for LeafSpritePool {
    fn default() -> Self {
        Self {
            sprites: Vec::new(),
            available: Vec::new(),
            active: Vec::new(),
        }
    }
}


fn tree_info_display(
    mut commands: Commands,
    tree_query: Query<(&Transform, &PixelTree)>,
    camera_query: Query<&Transform, (With<Camera>, Without<PixelTree>)>,
    spatial_index: Res<TreeSpatialIndex>,
    existing_text: Query<Entity, With<TreeInfoText>>,
    selected_tree: Res<SelectedTree>,
    shadow_settings: Res<ShadowSettings>,
    fps_counter: Res<FpsCounter>,
    mut last_update_timer: Local<f32>,
    time: Res<Time>,
) {
    // Only update UI every 0.1 seconds for better performance
    *last_update_timer += time.delta_secs();
    if *last_update_timer < 0.1 {
        return;
    }
    *last_update_timer = 0.0;
    
    // Clean up existing text
    for entity in existing_text.iter() {
        commands.entity(entity).despawn();
    }
    
    if let Ok(camera_transform) = camera_query.single() {
        let camera_pos = camera_transform.translation.truncate();
        let view_bounds = Rect::from_center_size(camera_pos, Vec2::new(1200.0, 800.0));
        
        let visible_trees = spatial_index.get_visible_trees(view_bounds);
        let total_trees = tree_query.iter().count();
        
        // Count trees by LOD level
        let mut lod_counts = [0; 4];
        for (_, tree) in tree_query.iter() {
            lod_counts[tree.lod_level as usize] += 1;
        }
        
        // Get wind info from first tree for display
        let wind_info = if let Some((_, tree)) = tree_query.iter().next() {
            format!(
                "Wind: Str:{:.1} Freq:{:.1} Turb:{:.1}\n\
                 Dir:({:.1}, {:.1})",
                tree.wind_params.strength,
                tree.wind_params.frequency, 
                tree.wind_params.turbulence,
                tree.wind_params.direction.x,
                tree.wind_params.direction.y
            )
        } else {
            "Wind: No trees".to_string()
        };

        let selection_info = if let Some(_) = selected_tree.entity {
            format!("Selected Tree: {:.0}m away", selected_tree.distance)
        } else {
            "No tree selected".to_string()
        };
        
        let shadow_info = format!("Shadows: {}", if shadow_settings.enabled { "ON" } else { "OFF" });

        let info_text = format!(
            "🌳 PixelTree Demo | FPS: {:.1}\n\
             Total Trees: {}\n\
             Visible: {}\n\
             LOD Levels: H:{} M:{} L:{} Min:{}\n\
             {}\n\
             {}\n\
             {}\n\
             \n\
             Controls:\n\
             WASD/Arrows: Move camera\n\
             Mouse: Drag to move\n\
             Z/X or Wheel: Zoom\n\
             R/F: Camera tilt\n\
             G: Regenerate selected tree\n\
             S: Toggle shadows\n\
             O: Optimize selected tree\n\
             \n\
             Camera: ({:.0}, {:.0})",
            fps_counter.fps,
            total_trees,
            visible_trees.len(),
            lod_counts[0], lod_counts[1], lod_counts[2], lod_counts[3],
            wind_info,
            selection_info,
            shadow_info,
            camera_pos.x, camera_pos.y
        );
        
        commands.spawn((
            Text::new(info_text),
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(10.0),
                top: Val::Px(10.0),
                ..default()
            },
            TreeInfoText,
        ));
    }
}

fn render_trees(
    mut gizmos: Gizmos,
    tree_query: Query<(&Transform, &PixelTree)>,
    camera_query: Query<(&Transform, &CameraZoom), With<Camera>>,
) {
    if let Ok((camera_transform, zoom)) = camera_query.single() {
        let camera_pos = camera_transform.translation.truncate();
        let view_distance = 1500.0 / zoom.0; // Adjust culling based on zoom (inverted)
        
        for (tree_transform, tree) in tree_query.iter() {
            let tree_pos = tree_transform.translation.truncate();
            
            // Skip trees too far from camera for performance - dynamic distance based on zoom
            if camera_pos.distance(tree_pos) > view_distance {
                continue;
            }
            
            // Render trunk segments
            let trunk_color = Color::srgb(0.4, 0.2, 0.1); // Brown
            for segment in &tree.trunk.segments {
                let start = tree_pos + segment.start;
                let end = tree_pos + segment.end;
                
                // Draw trunk as thick lines
                for i in 0..(segment.width as i32) {
                    let offset = i as f32 - segment.width * 0.5;
                    gizmos.line_2d(
                        start + Vec2::new(offset, 0.0),
                        end + Vec2::new(offset, 0.0),
                        trunk_color
                    );
                }
            }
            
            // Branches are now rendered as sprites for proper Z-ordering
            // Only render debug info if no branches exist
            if tree.branches.is_empty() {
                println!("WARNING: Tree at {:?} has NO branches!", tree_pos);
            }
            
            // No gizmo leaf rendering - all leaves are sprites
            
            // Draw tree base circle for debugging
            gizmos.circle_2d(tree_pos, 5.0, Color::srgba(1.0, 1.0, 1.0, 0.3));
        }
    }
}

fn setup_leaf_sprite_pool(
    mut commands: Commands,
    tree_query: Query<(Entity, &Transform, &PixelTree)>,
    mut sprite_pool: ResMut<LeafSpritePool>,
) {
    println!("Setting up optimized leaf sprite pool...");
    
    // Count total leaves across all trees
    let total_leaves: usize = tree_query.iter()
        .map(|(_, _, tree)| tree.leaves.leaves.len())
        .sum();
    
    println!("Creating sprite pool for {} total leaves", total_leaves);
    
    // Pre-allocate sprite pool with all needed sprites
    sprite_pool.sprites.reserve(total_leaves);
    sprite_pool.available.reserve(total_leaves);
    sprite_pool.active.reserve(total_leaves);
    
    // Create all leaf sprites at once for better batching
    for (sprite_index, (tree_entity, tree_transform, tree)) in tree_query.iter().enumerate() {
        let tree_pos = tree_transform.translation.truncate();
        
        for leaf in &tree.leaves.leaves {
            let leaf_pos = tree_pos + leaf.pos;
            
            // Add lighting variation based on leaf position and random factors (restored)
            let variation = (leaf.pos.x * 0.01 + leaf.pos.y * 0.01 + leaf.rot).sin() * 0.3 + 0.7;
            let shadow_factor = (leaf.pos.y * 0.005).cos() * 0.2 + 0.8; // Vertical gradient
            
            let base_green = (leaf.color[1] as f32 / 255.0).max(0.6);
            let leaf_color = Color::srgb(
                ((leaf.color[0] as f32 / 255.0) * variation * 0.8).max(0.1),
                (base_green * variation * shadow_factor).clamp(0.4, 1.0),
                ((leaf.color[2] as f32 / 255.0) * variation * 0.6).max(0.1),
            );
            
            let scale = (leaf.scale * 8.0).max(5.0);
            
            // Create sprite with consistent properties for better batching
            let sprite_entity = commands.spawn((
                Sprite {
                    color: leaf_color,
                    ..default()
                },
                Transform {
                    translation: leaf_pos.extend(20.0),
                    rotation: Quat::from_rotation_z(leaf.rot.to_radians()),
                    scale: Vec3::splat(scale),
                },
                LeafSprite {
                    tree_entity,
                    base_position: leaf.pos,
                    base_rotation: leaf.rot,
                },
            )).id();
            
            let pool_index = sprite_pool.sprites.len();
            sprite_pool.sprites.push(sprite_entity);
            sprite_pool.active.push(pool_index);
        }
    }
    
    println!("Created {} leaf sprites in pool", sprite_pool.sprites.len());
}

fn setup_branch_sprites(
    mut commands: Commands,
    tree_query: Query<(Entity, &Transform, &PixelTree)>,
) {
    println!("Setting up branch sprites once at startup...");
    
    // Create branch sprites for all trees once at startup  
    for (_tree_entity, tree_transform, tree) in tree_query.iter() {
        let tree_pos = tree_transform.translation.truncate();
        
        // Create sprites for each branch
        for branch in &tree.branches {
            let start = tree_pos + branch.start;
            let end = tree_pos + branch.end;
            
            let center = (start + end) / 2.0;
            let length = start.distance(end);
            let angle = (end - start).angle_to(Vec2::X);
            
            let branch_color = Color::srgb(0.6, 0.3, 0.1);
            let thickness = branch.thickness.max(2.0);
            
            // Create a stretched rectangle sprite for each branch
            commands.spawn((
                Sprite {
                    color: branch_color,
                    ..default()
                },
                Transform {
                    translation: center.extend(5.0), // Lower Z than leaves (20.0)
                    rotation: Quat::from_rotation_z(angle),
                    scale: Vec3::new(length, thickness, 1.0),
                },
                BranchSprite,
            ));
        }
    }
}

fn animate_leaves_and_branches(
    time: Res<Time>,
    tree_query: Query<(Entity, &Transform, &PixelTree)>,
    mut leaf_query: Query<(&mut Transform, &LeafSprite), Without<PixelTree>>,
    camera_query: Query<&Transform, (With<Camera>, Without<PixelTree>, Without<LeafSprite>)>,
    mut animation_timer: Local<f32>,
) {
    // Only update animation every few frames for better performance
    *animation_timer += time.delta_secs();
    if *animation_timer < 0.033 { // ~30 FPS animation updates
        return;
    }
    *animation_timer = 0.0;
    
    let elapsed = time.elapsed_secs();
    
    // Get camera position for culling
    let camera_pos = if let Ok(cam_transform) = camera_query.single() {
        cam_transform.translation.truncate()
    } else {
        Vec2::ZERO
    };
    
    // More aggressive distance culling for performance
    let max_animation_distance = 1200.0;
    
    // Pre-calculate common values
    let base_phase = elapsed * 2.4; // Combined frequency
    
    for (mut leaf_transform, leaf_sprite) in leaf_query.iter_mut() {
        // Find the corresponding tree
        if let Ok((_, tree_transform, tree)) = tree_query.get(leaf_sprite.tree_entity) {
            let tree_pos = tree_transform.translation.truncate();
            
            // Skip animation for distant trees
            if camera_pos.distance(tree_pos) > max_animation_distance {
                continue;
            }
            
            // Simplified wind calculation for better performance
            let wind_strength = tree.wind_params.strength * 0.4; // Reduced strength
            
            // Create phase variation with less computation
            let phase = base_phase + (leaf_sprite.base_position.x + leaf_sprite.base_position.y) * 0.005;
            
            // Simplified wind displacement
            let wind_x = phase.sin() * wind_strength;
            let wind_y = (phase * 0.7).cos() * wind_strength * 0.5;
            
            // Apply wind with less rotation calculation
            let wind_offset = Vec2::new(wind_x, wind_y);
            let wind_rotation = wind_x * 0.01; // Reduced rotational movement
            
            let new_pos = tree_pos + leaf_sprite.base_position + wind_offset;
            let new_rotation = leaf_sprite.base_rotation + wind_rotation;
            
            // Update the leaf transform
            leaf_transform.translation = new_pos.extend(20.0);
            leaf_transform.rotation = Quat::from_rotation_z(new_rotation.to_radians());
        }
    }
}

fn tree_selection_system(
    mut selected_tree: ResMut<SelectedTree>,
    camera_query: Query<&Transform, (With<Camera>, Without<PixelTree>)>,
    tree_query: Query<(Entity, &Transform), With<PixelTree>>,
    mut gizmos: Gizmos,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    mut spatial_index: ResMut<TreeSpatialIndex>,
    time: Res<Time>,
    mut shadow_settings: ResMut<ShadowSettings>,
) {
    if let Ok(camera_transform) = camera_query.single() {
        let camera_pos = camera_transform.translation.truncate();
        
        // Find the tree closest to camera center (viewport center)
        let mut closest_entity = None;
        let mut closest_distance = f32::MAX;
        
        for (entity, tree_transform) in tree_query.iter() {
            let tree_pos = tree_transform.translation.truncate();
            let distance = camera_pos.distance(tree_pos);
            
            if distance < closest_distance {
                closest_distance = distance;
                closest_entity = Some(entity);
            }
        }
        
        // Update selected tree
        selected_tree.entity = closest_entity;
        selected_tree.distance = closest_distance;
        
        // Draw selection indicator around the closest tree
        if let Some(selected_entity) = selected_tree.entity {
            if let Ok((_, tree_transform)) = tree_query.get(selected_entity) {
                let tree_pos = tree_transform.translation.truncate();
                
                // Draw pulsing selection circle
                let time_pulse = (time.elapsed_secs() * 3.0).sin() * 0.5 + 0.5;
                let radius = 30.0 + time_pulse * 10.0;
                gizmos.circle_2d(tree_pos, radius, Color::srgba(1.0, 1.0, 0.0, 0.8)); // Bright yellow
                
                // Draw crosshair at center
                gizmos.line_2d(
                    tree_pos + Vec2::new(-15.0, 0.0),
                    tree_pos + Vec2::new(15.0, 0.0),
                    Color::srgb(1.0, 1.0, 0.0)
                );
                gizmos.line_2d(
                    tree_pos + Vec2::new(0.0, -15.0),
                    tree_pos + Vec2::new(0.0, 15.0),
                    Color::srgb(1.0, 1.0, 0.0)
                );
            }
        }
        
        // Handle shadow toggle (S key)
        if keyboard.just_pressed(KeyCode::KeyS) {
            shadow_settings.enabled = !shadow_settings.enabled;
            println!("Shadows {}", if shadow_settings.enabled { "enabled" } else { "disabled" });
        }
        
        // Handle tree optimization (O key)
        if keyboard.just_pressed(KeyCode::KeyO) {
            if let Some(selected_entity) = selected_tree.entity {
                if let Ok((_, tree_transform)) = tree_query.get(selected_entity) {
                    let tree_pos = tree_transform.translation.truncate();
                    println!("Optimizing tree at {:?}", tree_pos);
                    commands.spawn(OptimizeTreeMarker { entity: selected_entity });
                }
            }
        }
        
        // Handle regeneration key (G for Generate)
        if keyboard.just_pressed(KeyCode::KeyG) {
            if let Some(selected_entity) = selected_tree.entity {
                if let Ok((_, tree_transform)) = tree_query.get(selected_entity) {
                    let tree_pos = tree_transform.translation.truncate();
                    println!("Regenerating tree at {:?}", tree_pos);
                    
                    // Despawn the old tree and associated sprites
                    commands.entity(selected_entity).despawn();
                    
                    // Create new random tree at the same position
                    let params = GenerationParams {
                        seed: (time.elapsed_secs() * 1000.0) as u64, // Random seed based on elapsed time
                        height_range: (80.0, 140.0),
                        wind_params: WindParams {
                            strength: 2.0 + (tree_pos.x * 0.001).sin() * 0.5,
                            frequency: 1.2 + (tree_pos.y * 0.001).cos() * 0.3,
                            turbulence: 0.4,
                            direction: Vec2::new(1.0, 0.2),
                        },
                        branch_angle_variance: 45.0,
                        leaf_density: 3.0,
                        lod_level: 0,
                        ..default()
                    };
                    
                    // Spawn new tree and immediately create sprites for it
                    let new_tree_entity = spawn_tree(&mut commands, tree_pos.extend(0.0), params, &mut spatial_index);
                    
                    // We need to trigger sprite regeneration in the next frame since the tree data isn't available yet
                    commands.spawn(RegeneratedTreeMarker { entity: new_tree_entity });
                    println!("New tree spawned!");
                }
            }
        }
    }
}

fn handle_regenerated_trees(
    mut commands: Commands,
    tree_query: Query<(Entity, &Transform, &PixelTree)>,
    marker_query: Query<(Entity, &RegeneratedTreeMarker)>,
    mut leaf_sprite_query: Query<(&mut Transform, &mut LeafSprite, &mut Sprite), Without<PixelTree>>,
    branch_sprite_query: Query<Entity, With<BranchSprite>>,
) {
    // Handle trees that were regenerated and need sprites
    for (marker_entity, marker) in marker_query.iter() {
        if let Ok((tree_entity, tree_transform, tree)) = tree_query.get(marker.entity) {
            let tree_pos = tree_transform.translation.truncate();
            
            println!("Efficiently updating sprites for regenerated tree at {:?}", tree_pos);
            
            // Update existing sprites for this tree with new data
            let mut updated_sprites = 0;
            for (mut transform, mut leaf_sprite, mut sprite) in leaf_sprite_query.iter_mut() {
                if leaf_sprite.tree_entity == marker.entity {
                    // This sprite belongs to the regenerated tree
                    if updated_sprites < tree.leaves.leaves.len() {
                        let leaf = &tree.leaves.leaves[updated_sprites];
                        let leaf_pos = tree_pos + leaf.pos;
                        
                        // Calculate proper lighting
                        let variation = (leaf.pos.x * 0.01 + leaf.pos.y * 0.01 + leaf.rot).sin() * 0.3 + 0.7;
                        let shadow_factor = (leaf.pos.y * 0.005).cos() * 0.2 + 0.8;
                        
                        let base_green = (leaf.color[1] as f32 / 255.0).max(0.6);
                        let leaf_color = Color::srgb(
                            ((leaf.color[0] as f32 / 255.0) * variation * 0.8).max(0.1),
                            (base_green * variation * shadow_factor).clamp(0.4, 1.0),
                            ((leaf.color[2] as f32 / 255.0) * variation * 0.6).max(0.1),
                        );
                        
                        let scale = (leaf.scale * 8.0).max(5.0);
                        
                        // Update the existing sprite
                        transform.translation = leaf_pos.extend(20.0);
                        transform.rotation = Quat::from_rotation_z(leaf.rot.to_radians());
                        transform.scale = Vec3::splat(scale);
                        sprite.color = leaf_color;
                        leaf_sprite.base_position = leaf.pos;
                        leaf_sprite.base_rotation = leaf.rot;
                        leaf_sprite.tree_entity = tree_entity; // Update to new tree entity
                        
                        updated_sprites += 1;
                    } else {
                        // More existing sprites than new leaves, hide this sprite
                        transform.scale = Vec3::ZERO; // Hide unused sprite
                    }
                }
            }
            
            // If we need more sprites than existing ones, create new ones
            for i in updated_sprites..tree.leaves.leaves.len() {
                let leaf = &tree.leaves.leaves[i];
                let leaf_pos = tree_pos + leaf.pos;
                
                let variation = (leaf.pos.x * 0.01 + leaf.pos.y * 0.01 + leaf.rot).sin() * 0.3 + 0.7;
                let shadow_factor = (leaf.pos.y * 0.005).cos() * 0.2 + 0.8;
                
                let base_green = (leaf.color[1] as f32 / 255.0).max(0.6);
                let leaf_color = Color::srgb(
                    ((leaf.color[0] as f32 / 255.0) * variation * 0.8).max(0.1),
                    (base_green * variation * shadow_factor).clamp(0.4, 1.0),
                    ((leaf.color[2] as f32 / 255.0) * variation * 0.6).max(0.1),
                );
                
                let scale = (leaf.scale * 8.0).max(5.0);
                
                commands.spawn((
                    Sprite {
                        color: leaf_color,
                        ..default()
                    },
                    Transform {
                        translation: leaf_pos.extend(20.0),
                        rotation: Quat::from_rotation_z(leaf.rot.to_radians()),
                        scale: Vec3::splat(scale),
                    },
                    LeafSprite {
                        tree_entity,
                        base_position: leaf.pos,
                        base_rotation: leaf.rot,
                    },
                ));
            }
            
            // Despawn old branch sprites and create new ones
            for entity in branch_sprite_query.iter() {
                commands.entity(entity).despawn();
            }
            
            // Create branch sprites for the new tree
            for branch in &tree.branches {
                let start = tree_pos + branch.start;
                let end = tree_pos + branch.end;
                let center = (start + end) / 2.0;
                let length = start.distance(end);
                let angle = (end - start).angle_to(Vec2::X);
                
                let branch_color = Color::srgb(0.6, 0.3, 0.1);
                let thickness = branch.thickness.max(2.0);
                
                commands.spawn((
                    Sprite {
                        color: branch_color,
                        ..default()
                    },
                    Transform {
                        translation: center.extend(5.0),
                        rotation: Quat::from_rotation_z(angle),
                        scale: Vec3::new(length, thickness, 1.0),
                    },
                    BranchSprite,
                ));
            }
        }
        
        // Remove the marker since we've handled it
        commands.entity(marker_entity).despawn();
    }
}

fn shadow_system(
    mut commands: Commands,
    shadow_settings: Res<ShadowSettings>,
    tree_query: Query<(Entity, &Transform, &PixelTree)>,
    existing_shadows: Query<Entity, With<ShadowCaster>>,
) {
    // If shadows changed state, update all shadows
    if shadow_settings.is_changed() {
        // Remove existing shadows
        for shadow_entity in existing_shadows.iter() {
            commands.entity(shadow_entity).despawn();
        }
        
        // Add new shadows if enabled
        if shadow_settings.enabled {
            for (_tree_entity, tree_transform, tree) in tree_query.iter() {
                let tree_pos = tree_transform.translation.truncate();
                
                // Create a simple circular shadow underneath the tree
                let shadow_offset = Vec2::new(15.0, -25.0); // Offset shadow to bottom-right
                let shadow_pos = tree_pos + shadow_offset;
                
                // Create shadow sprite
                commands.spawn((
                    Sprite {
                        color: Color::srgba(0.0, 0.0, 0.0, 0.3), // Semi-transparent black
                        ..default()
                    },
                    Transform {
                        translation: shadow_pos.extend(0.1), // Very low Z to render under everything
                        scale: Vec3::new(
                            tree.trunk.height * 0.8, // Shadow width based on tree height
                            tree.trunk.height * 0.4, // Shadow height (flattened)
                            1.0
                        ),
                        ..default()
                    },
                    ShadowCaster,
                ));
                
                // Add smaller shadows for major branches
                for branch in tree.branches.iter().take(5) { // Only first 5 branches for performance
                    if branch.thickness > 3.0 { // Only thick branches cast noticeable shadows
                        let branch_center = tree_pos + (branch.start + branch.end) / 2.0;
                        let branch_shadow_pos = branch_center + shadow_offset * 0.5;
                        
                        commands.spawn((
                            Sprite {
                                color: Color::srgba(0.0, 0.0, 0.0, 0.15),
                                ..default()
                            },
                            Transform {
                                translation: branch_shadow_pos.extend(0.05),
                                scale: Vec3::new(
                                    branch.start.distance(branch.end) * 0.6,
                                    branch.thickness * 0.8,
                                    1.0
                                ),
                                ..default()
                            },
                            ShadowCaster,
                        ));
                    }
                }
            }
            println!("Created shadows for {} trees", tree_query.iter().count());
        } else {
            println!("Removed all shadows");
        }
    }
}

fn fps_system(
    mut fps_counter: ResMut<FpsCounter>,
    time: Res<Time>,
) {
    fps_counter.frame_count += 1;
    fps_counter.timer += time.delta_secs();
    
    // Update FPS every second
    if fps_counter.timer >= 1.0 {
        fps_counter.fps = fps_counter.frame_count as f32 / fps_counter.timer;
        fps_counter.frame_count = 0;
        fps_counter.timer = 0.0;
    }
}

fn tree_optimization_system(
    mut commands: Commands,
    mut tree_query: Query<&mut PixelTree>,
    marker_query: Query<(Entity, &OptimizeTreeMarker)>,
    _leaf_sprite_query: Query<Entity, With<LeafSprite>>,
    _branch_sprite_query: Query<Entity, With<BranchSprite>>,
) {
    for (marker_entity, marker) in marker_query.iter() {
        if let Ok(mut tree) = tree_query.get_mut(marker.entity) {
            let original_branch_count = tree.branches.len();
            let _original_leaf_count = tree.leaves.leaves.len();
            
            // Create a set of leaf positions for quick lookup
            let leaf_positions: std::collections::HashSet<_> = tree.leaves.leaves
                .iter()
                .map(|leaf| (leaf.pos.x as i32, leaf.pos.y as i32)) // Discretize positions
                .collect();
            
            // Remove branches that don't have leaves near them or point downward without purpose
            let mut branches_to_keep = Vec::new();
            for (i, branch) in tree.branches.iter().enumerate() {
                let branch_center = (branch.start + branch.end) / 2.0;
                let branch_direction = (branch.end - branch.start).normalize();
                let search_radius = 35.0; // Increased search radius for leaves
                
                // Check if any leaves are near this branch
                let has_nearby_leaves = leaf_positions.iter().any(|(leaf_x, leaf_y)| {
                    let leaf_pos = Vec2::new(*leaf_x as f32, *leaf_y as f32);
                    branch_center.distance(leaf_pos) <= search_radius
                });
                
                // Check if branch points significantly downward (orphaned downward branches)
                let points_downward = branch_direction.y < -0.5;
                
                // Keep branches if:
                // 1. Main structural branches (generation 0-1)
                // 2. Has nearby leaves
                // 3. Thick branches (structural importance)
                // 4. NOT pointing downward without leaves
                let should_keep = (branch.generation <= 1) || 
                                 (has_nearby_leaves) || 
                                 (branch.thickness > 3.0) ||
                                 (!points_downward);
                
                if should_keep && !(points_downward && !has_nearby_leaves) {
                    branches_to_keep.push(i);
                }
            }
            
            // Filter the branches
            let mut new_branches = Vec::new();
            for &index in &branches_to_keep {
                new_branches.push(tree.branches[index].clone());
            }
            tree.branches = new_branches;
            
            // Don't remove leaves - user wants to keep all leaves
            let new_branch_count = tree.branches.len();
            let leaf_count = tree.leaves.leaves.len();
            
            println!("Tree optimized: Branches: {} -> {} ({:.1}% reduction), Leaves: {} (unchanged)",
                original_branch_count, new_branch_count,
                (original_branch_count - new_branch_count) as f32 / original_branch_count as f32 * 100.0,
                leaf_count
            );
            
            // Trigger sprite regeneration for the optimized tree
            commands.spawn(RegeneratedTreeMarker { entity: marker.entity });
        }
        
        // Remove the marker
        commands.entity(marker_entity).despawn();
    }
}

fn force_high_lod(mut tree_query: Query<&mut PixelTree>) {
    // Force all trees to stay at LOD level 0 (high detail) for demo
    for mut tree in tree_query.iter_mut() {
        if tree.lod_level != 0 {
            tree.lod_level = 0;
        }
    }
}
