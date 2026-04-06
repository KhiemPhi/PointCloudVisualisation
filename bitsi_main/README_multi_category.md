# Multi-Category Scene Generation

This script generates scenes with objects from different categories without performing any segmentation. It simply loads and places objects in a scene for visualization and analysis.

## Features

- **Multi-category scenes**: Places objects from different categories in a scene
- **No segmentation**: Objects are loaded as-is without any processing
- **Dataset support**: Works with all datasets (HANDAL, YCBV, KITTI, ShapeNetPart)
- **Random positioning**: Objects are placed at random positions and orientations
- **Flexible object selection**: Can specify exact objects or load random ones

## Usage

### Basic Usage

```bash
# Generate a scene with specific objects from YCBV dataset
python main_multi_category.py --parent_dir YCBV --objects toy hammer screwdriver

# Generate a scene with 5 random objects from ShapeNetPart
python main_multi_category.py --parent_dir ShapeNetPart --num_objects 5

# Generate a scene with specific objects from HANDAL dataset
python main_multi_category.py --parent_dir HANDAL --objects hammer wrench screwdriver
```

### Advanced Usage

```bash
# Custom scene size
python main_multi_category.py \
    --parent_dir YCBV \
    --objects toy hammer screwdriver \
    --scene_size 2.5

# Skip individual object visualization
python main_multi_category.py \
    --parent_dir ShapeNetPart \
    --objects airplane chair table \
    --no_individual
```

## Parameters

### Dataset Parameters
- `--parent_dir`: Dataset directory (HANDAL, YCBV, YCBV-Partial, KITTI, ShapeNetPart)
- `--objects`: List of specific object categories to load
- `--num_objects`: Number of random objects to load (alternative to --objects)

### Scene Parameters
- `--scene_size`: Size of the scene radius (default: 2.0)

### Visualization Parameters
- `--no_individual`: Skip individual object visualization

## Examples

### Example 1: YCBV Dataset with Specific Objects
```bash
python main_multi_category.py --parent_dir YCBV --objects toy hammer screwdriver
```
- Loads 3 specific objects from YCBV dataset
- Places them at random positions in a 2.0 radius scene
- Shows individual objects and combined scene

### Example 2: ShapeNetPart Dataset with Random Objects
```bash
python main_multi_category.py --parent_dir ShapeNetPart --num_objects 4
```
- Loads 4 random objects from ShapeNetPart dataset
- Each object is from a different category
- Shows the diversity of the dataset

### Example 3: HANDAL Dataset with Tools
```bash
python main_multi_category.py --parent_dir HANDAL --objects hammer wrench screwdriver
```
- Loads 3 different tools from HANDAL dataset
- Each tool is placed at random positions
- Suitable for tool manipulation scenarios

### Example 4: KITTI Dataset with Random Objects
```bash
python main_multi_category.py --parent_dir KITTI --num_objects 2 --scene_size 3.0
```
- Loads 2 random car objects from KITTI dataset
- Places them in a larger 3.0 radius scene
- Suitable for autonomous driving scenarios

## Object Selection

### Specified Objects (`--objects`)
- **YCBV**: `toy`, `hammer`, `screwdriver`, `wrench`, `pliers`, etc.
- **ShapeNetPart**: `airplane`, `chair`, `table`, `car`, `lamp`, etc.
- **HANDAL**: `hammer`, `wrench`, `screwdriver`, `pliers`, etc.
- **KITTI**: `car`, `truck`, `bus`, etc.

### Random Objects (`--num_objects`)
- Automatically selects random objects from the dataset
- Ensures diversity in the scene
- Useful for exploring dataset contents

## Scene Layout

Objects are placed using:
- **Random positions**: Within a circular area of specified radius
- **Random rotations**: Around the Z-axis (0 to 2π)
- **Random scales**: Slight size variations (0.8 to 1.2x)
- **Height variation**: Small Z-axis offsets (-0.2 to 0.2)

## Output

The script generates:

1. **Individual Object Visualization**: Shows each object with different colors
2. **Combined Scene Visualization**: Shows the merged point cloud
3. **Object Information**: Prints details about each loaded object
4. **Scene Statistics**: Total points, scene bounds, etc.

## Dataset-Specific Behavior

### YCBV/YCBV-Partial
- Loads objects from `/home/khiem/Robotics/obj-decomposition/YCBV/`
- Each object is from a different category
- Objects are scaled and positioned randomly

### ShapeNetPart
- Loads random objects from different categories
- Each object has different part annotations
- Shows the diversity of the dataset

### HANDAL
- Loads objects with specific object numbers
- Objects are transformed and positioned randomly
- Suitable for tool/object manipulation scenarios

### KITTI
- Loads car objects from KITTI dataset
- Objects are scaled appropriately for the scene
- Suitable for autonomous driving scenarios

## Performance

### Processing Time
- **Small scenes (2-3 objects)**: ~10-30 seconds
- **Medium scenes (4-6 objects)**: ~30-60 seconds
- **Large scenes (7+ objects)**: ~1-2 minutes

### Memory Usage
- **Per object**: ~50-200 MB depending on point cloud size
- **Combined scene**: ~200-1000 MB depending on number of objects
- **Peak usage**: ~500 MB during processing

## Use Cases

### Dataset Exploration
- **Explore dataset contents**: See what objects are available
- **Visualize diversity**: Understand the range of objects in a dataset
- **Quality assessment**: Check object quality and completeness

### Scene Understanding
- **Multi-object scenarios**: Understand how objects interact
- **Spatial relationships**: Analyze object placement and orientation
- **Scene complexity**: Assess scene complexity for algorithms

### Algorithm Testing
- **Multi-object algorithms**: Test algorithms on multi-object scenes
- **Object detection**: Test object detection on complex scenes
- **Scene understanding**: Test scene understanding algorithms

### Visualization
- **Dataset visualization**: Show dataset contents visually
- **Scene composition**: Understand scene composition
- **Object relationships**: Analyze spatial relationships

## Troubleshooting

### Common Issues

1. **"No objects found for dataset"**
   - Check dataset path
   - Verify dataset is properly installed
   - Check file permissions

2. **"Failed to load [object_name]"**
   - Check if object exists in dataset
   - Verify object name spelling
   - Check file permissions

3. **"No objects loaded successfully"**
   - Check dataset path
   - Verify object names
   - Check file permissions

### Debug Mode

For debugging, add print statements:

```python
# In main_multi_category.py
print(f"Available objects: {available_objects}")
print(f"Loading object: {object_name}")
print(f"Object info: {object_info}")
```

## Integration

The multi-category scene generation can be integrated with:

1. **Dataset exploration**: Understand dataset contents
2. **Algorithm testing**: Test algorithms on multi-object scenes
3. **Scene understanding**: Analyze multi-object scenarios
4. **Visualization**: Show dataset diversity

## Next Steps

1. **Custom layouts**: Implement specific object placement patterns
2. **Collision detection**: Add object collision avoidance
3. **Physics simulation**: Integrate with physics engines
4. **Real-time processing**: Optimize for real-time applications
