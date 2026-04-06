import numpy as np
from scipy.ndimage import label
from scipy.ndimage import label, generate_binary_structure

class OccupancyStruct:
    def __init__(self, bounding_box, point_cloud, resolution, expansion=3):
        self.bounding_box = np.array(bounding_box, dtype=np.float64)
        self.point_cloud = np.asarray(point_cloud, dtype=np.float64)
        self.resolution = resolution
        self.expansion = expansion

        self.occupancy_grid = self.construct_occupancy_grid()
        self.shape = self.occupancy_grid.shape

        # Connected components using SciPy (MUCH faster than NetworkX)
        occupied_labels, num_occ = label(self.occupancy_grid)
        empty_labels, num_empty = label(1 - self.occupancy_grid)

        self.occupied_comps = num_occ
        self.empty_comps = num_empty

        grid_volume = np.prod(self.shape)
        self.empty_factor = self.empty_comps / (self.empty_comps + self.occupied_comps)
        self.weighted_empty_percentage = self.empty_factor  # can scale with grid volume if desired
    
    def construct_occupancy_grid(self):
        x_min, y_min, z_min, x_max, y_max, z_max = self.bounding_box
        grid_width = int((x_max - x_min) / self.resolution) + self.expansion
        grid_height = int((y_max - y_min) / self.resolution) + self.expansion
        grid_depth = int((z_max - z_min) / self.resolution) + self.expansion

        grid = np.zeros((grid_width, grid_height, grid_depth), dtype=np.uint8)

        # Vectorized point → grid index conversion
        coords = ((self.point_cloud - np.array([x_min, y_min, z_min])) / self.resolution).astype(int)
        valid = np.all((coords >= 0) & (coords < np.array([grid_width, grid_height, grid_depth])), axis=1)
        coords = coords[valid]

        grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        return grid
    

    def count_connected_components(self, grid, occupied=1):
        structure = generate_binary_structure(rank=3, connectivity=3)
        labeled, n_components = label(grid == occupied, structure)
        return n_components
