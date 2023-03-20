# standard libraries
import math
from typing import Tuple

# third party libraries
import numpy as np
import numpy.typing as npt
import pyvista as pv
import typer


def vase_scalar_field(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculate values in a scalar field whose isosurface at zero defines the wall of a vase."""
    xy, z = points[:, :2], points[:, 2]
    return -np.linalg.norm(xy, axis=1) + 4.0 * ((z >= 1) & (z <= 9))


def _conf_axis(resolution: float, length: float, center: bool) -> Tuple[float, float, float]:
    div = math.ceil(length / resolution)
    num = div + 1
    start = (-length / 2) if center else 0
    step = length / div
    return num, step, start


def build_grid(
    build_volume: Tuple[float, float, float],
    xy_resolution: float,
    z_resolution: float,
    extra_resolution_factor: float,
) -> pv.UniformGrid:
    """Build a grid from 3D printer specs."""
    xy_resolution /= extra_resolution_factor
    z_resolution /= extra_resolution_factor

    dimensions, spacing, origin = zip(
        _conf_axis(xy_resolution, build_volume[0], center=True),
        _conf_axis(xy_resolution, build_volume[1], center=True),
        _conf_axis(z_resolution, build_volume[2], center=False),
    )

    return pv.UniformGrid(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin,
    )


def main(out_file: str = "vase.stl") -> None:
    """Create a mesh for a printable vase and save as an stl."""
    grid = build_grid(
        build_volume=(10.0, 10.0, 10.0),
        xy_resolution=0.5,
        z_resolution=0.5,
        extra_resolution_factor=10.0,
    )
    values = vase_scalar_field(grid.points)
    mesh = grid.contour([0], values, method="marching_cubes")
    # todo: decimate the mesh
    mesh.save(out_file)


if __name__ == "__main__":
    typer.run(main)
