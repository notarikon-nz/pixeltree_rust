# SpeedTree STF/SRT Import Feature

This document describes the SpeedTree import functionality added to PixelTree.

## Overview

PixelTree now supports importing SpeedTree forest layouts via STF (SpeedTree Forest) files and has a foundation for importing tree geometry from SRT/ST files.

## Features Implemented

### STF Forest Loading
- **File Format**: ASCII text files containing tree positions, rotations, scales, and model references
- **Parser**: Robust STF file parser with comprehensive error handling
- **Integration**: Seamless integration with PixelTree's procedural generation system
- **Batch Processing**: Parallel tree generation from STF data using Rayon

### SRT/ST Tree Import (Foundation)
- **Structure**: Basic binary parser framework for SRT/ST files
- **Data Extraction**: Framework for converting SpeedTree geometry to PixelTree format
- **LOD Support**: Multiple level-of-detail handling for imported trees
- **Future Extensibility**: Ready for binary format reverse engineering

## New API Functions

### High-Level Functions
```rust
// Load and spawn forest from STF file
spawn_forest_from_stf(commands, "forest.stf", params, spatial_index) -> Result<Vec<Entity>, ImportError>

// Load and spawn individual SRT/ST tree
spawn_imported_tree(commands, "tree.srt", position, wind_params, lod_level, spatial_index) -> Result<Entity, ImportError>
```

### Batch Processing
```rust
let batch_gen = BatchTreeGenerator::new(base_params);

// Generate forest from STF file
let trees = batch_gen.generate_forest_from_stf("forest.stf")?;

// Mix procedural and imported trees
let mixed_forest = batch_gen.generate_mixed_forest(procedural_positions, imported_trees, imported_positions)?;
```

### Direct STF Loading
```rust
// Parse STF file directly
let forest_layout = StfForestLoader::load_stf("forest.stf")?;

// Spawn trees from parsed layout
let entities = StfForestLoader::spawn_from_stf(commands, &forest_layout, params, spatial_index);
```

## STF File Format

The STF format is a simple ASCII text file:

```
TreeFilename.srt
[Number of Instances]
x y z rotation scale
x y z rotation scale
...
NextTreeFilename.srt
[Number of Instances]
x y z rotation scale
...
```

### Example STF File
```
oak_tree_01.srt
3
100.0 0.0 0.0 0.0 1.2
-50.0 0.0 0.0 0.523599 0.9
50.0 80.0 0.0 -0.261799 1.1
```

## Integration with Existing Systems

### Deterministic Generation
- STF trees maintain deterministic generation using filename and position hashing
- Different tree files produce varied but consistent results
- Scale and rotation from STF files influence tree parameters

### Spatial Indexing
- Imported trees are automatically added to the spatial index
- Full compatibility with existing culling and LOD systems
- Performance optimizations through viewport culling

### Wind Animation
- STF rotation influences wind direction for natural variation
- Scale affects tree dimensions and wind response
- Full compatibility with existing wind animation system

## Demo Integration

The demo (`cargo run --example demo`) now includes:
- Automatic STF loading on startup from `examples/sample_forest.stf`
- **L key**: Reload STF forest with different parameters
- Visual indicators for imported vs procedural trees
- Real-time performance metrics including imported tree counts

## Testing

Comprehensive test suite covers:
- STF parsing with valid and invalid inputs
- Multiple tree type handling
- Batch forest generation from STF files  
- SRT file validation and parsing
- Tree creation from imported data with different LOD levels
- Deterministic hashing for tree variation

## Future Enhancements

### SRT/ST Binary Format
The current SRT/ST implementation is a placeholder. Future enhancements could include:
- Reverse engineering of SpeedTree binary formats
- Direct geometry extraction from SRT/ST files
- Material and texture mapping support
- Advanced LOD level definitions from SpeedTree data

### Performance Optimizations
- Memory-mapped file loading for large STF files
- Streaming import for massive forests
- GPU-accelerated geometry processing
- Instancing support for repeated tree models

## Dependencies

Added dependencies:
- `nom = "7.1"` - For potential future binary parsing (currently unused)
- `thiserror = "2.0"` - For robust error handling

## Error Handling

Comprehensive error handling with `ImportError` enum:
- `FileError` - File system errors
- `ParseError` - STF parsing errors with detailed messages
- `UnsupportedFormat` - Format validation errors
- `InvalidData` - Data validation errors

All functions return `Result<T, ImportError>` for proper error propagation.