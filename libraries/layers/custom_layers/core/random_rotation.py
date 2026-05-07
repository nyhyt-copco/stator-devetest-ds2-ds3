"""Defines layer for randomly rotating mesh or point cloud data using numpy rotation matrices."""

from typing import Literal

import numpy as np

from samples.data_blocks.point_cloud import PointCloud
from samples.data_blocks.surface_mesh import SurfaceMesh
from samples.layers.core.base import Layer, SampleT


class RandomRotation(Layer):
    """Randomly rotates mesh or point cloud data using efficient numpy rotation matrices.

    This layer applies random rotations to 3D mesh or point cloud data by directly
    manipulating the point coordinates using numpy arrays and rotation matrices,
    avoiding expensive PyVista rotation operations during training.
    """

    def __init__(
        self,
        input_datablock: str,
        *,
        rotation_axes: Literal["x", "y", "z", "xy", "xz", "yz", "xyz"] = "xyz",
        angle_range_degrees: tuple[float, float] = (0.0, 360.0),
        seed: int | None = None,
        tags: set[str] | None = None,
    ) -> None:
        """Initialize RandomRotation layer.

        Args:
            input_datablock (str): Name of the SurfaceMesh or PointCloud datablock to rotate.
            rotation_axes (Literal["x", "y", "z", "xy", "xz", "yz", "xyz"], optional):
                Axes around which to apply random rotations. Single axis rotations ("x", "y", "z")
                apply rotation only around that axis. Multiple axes ("xy", "xz", "yz", "xyz")
                apply combined rotations around the specified axes. Defaults to "xyz".
            angle_range_degrees (tuple[float, float], optional): Range of rotation angles in
                degrees as (min_angle, max_angle). Each axis gets a random angle within this range.
                Defaults to (0.0, 360.0).
            seed (int | None, optional): Random seed for reproducible rotations. If None,
                uses non-deterministic random state. Defaults to None.
            tags (set[str] | None, optional): Tags associated with this layer. These can be used to
                include or exclude particular layers from preprocessing when calling
                `Sample.apply()`. Defaults to None.

        Raises:
            ValueError: If angle_range_degrees min is greater than max, or if rotation_axes
                       is not a valid option.
        """
        super().__init__(tags=tags)

        # Validate inputs
        valid_axes = {"x", "y", "z", "xy", "xz", "yz", "xyz"}
        if rotation_axes not in valid_axes:
            raise ValueError(
                f"Invalid rotation_axes '{rotation_axes}'. Valid options are: {valid_axes}"
            )

        if angle_range_degrees[0] > angle_range_degrees[1]:
            min_angle, max_angle = angle_range_degrees
            raise ValueError(f"angle_range_degrees min ({min_angle}) must be <= max ({max_angle})")

        self.input_datablock = input_datablock
        self.rotation_axes = rotation_axes
        self.angle_range_degrees = angle_range_degrees
        self.seed = seed

        # Initialize random state
        self._rng = np.random.default_rng(seed)

    def _generate_rotation_matrix(self) -> np.ndarray:
        """Generate a random 3D rotation matrix based on the configured axes and angle range.

        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        # Get random angles for each axis
        min_angle, max_angle = self.angle_range_degrees

        # Convert to radians
        min_rad = np.deg2rad(min_angle)
        max_rad = np.deg2rad(max_angle)

        # Generate random angles for each axis
        angles = {}
        for axis in "xyz":
            if axis in self.rotation_axes:
                angles[axis] = self._rng.uniform(min_rad, max_rad)
            else:
                angles[axis] = 0.0

        # Create individual rotation matrices
        # Rotation around X-axis
        cos_x, sin_x = np.cos(angles["x"]), np.sin(angles["x"])
        r_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])

        # Rotation around Y-axis
        cos_y, sin_y = np.cos(angles["y"]), np.sin(angles["y"])
        r_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])

        # Rotation around Z-axis
        cos_z, sin_z = np.cos(angles["z"]), np.sin(angles["z"])
        r_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

        # Combine rotations: R = R_z * R_y * R_x
        # Order matters for combined rotations
        rotation_matrix = r_z @ r_y @ r_x

        return rotation_matrix

    def _rotate_points(self, points: np.ndarray) -> np.ndarray:
        """Rotate a set of 3D points using the generated rotation matrix.

        Args:
            points (np.ndarray): Array of 3D points with shape (n_points, 3)

        Returns:
            np.ndarray: Rotated points with same shape as input
        """
        if points.shape[1] != 3:
            raise ValueError(f"Points must have 3 columns (x, y, z), got shape {points.shape}")

        rotation_matrix = self._generate_rotation_matrix()

        # Apply rotation: rotated_points = points @ R.T
        # Using transpose because we want to rotate points, not coordinate system
        rotated_points = points @ rotation_matrix.T

        return rotated_points

    def _rotate_surface_mesh_datablock(self, datablock: SurfaceMesh) -> None:
        """Rotate points directly in a SurfaceMesh datablock.

        Args:
            datablock (SurfaceMesh): The surface mesh datablock to modify in place
        """
        # Read current geometry
        points, faces = datablock.read_geometry()

        # Rotate the points
        rotated_points = self._rotate_points(points)

        # Write the updated geometry back to the datablock
        datablock._write_geometry(rotated_points, faces)

    def _rotate_point_cloud_datablock(self, datablock: PointCloud) -> None:
        """Rotate points directly in a PointCloud datablock.

        Args:
            datablock (PointCloud): The point cloud datablock to modify in place
        """
        # Read current geometry
        points = datablock.read_geometry()

        # Rotate the points
        rotated_points = self._rotate_points(points)

        # Write the updated geometry back to the datablock
        datablock._write_geometry(rotated_points)

    def __call__(self, sample: SampleT) -> SampleT:
        """Apply random rotation to mesh or point cloud data.

        Args:
            sample (SampleT): Sample input.

        Returns:
            SampleT: Sample with randomly rotated data.

        Raises:
            RuntimeError: If the input datablock doesn't exist or is not a supported type.
        """
        # Get the datablock
        datablock = sample.get(self.input_datablock)
        if datablock is None:
            raise RuntimeError(f"Datablock '{self.input_datablock}' not found in sample")

        # Handle SurfaceMesh - directly modify the datablock
        if isinstance(datablock, SurfaceMesh):
            self._rotate_surface_mesh_datablock(datablock)

        # Handle PointCloud - directly modify the datablock
        elif isinstance(datablock, PointCloud):
            self._rotate_point_cloud_datablock(datablock)

        else:
            # Unsupported datablock type
            datablock_type = type(datablock).__name__
            raise RuntimeError(
                f"Datablock '{self.input_datablock}' is of unsupported type '{datablock_type}'. "
                f"Supported types are: SurfaceMesh, PointCloud"
            )
        return sample
