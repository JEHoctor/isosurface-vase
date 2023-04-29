# standard libraries
import itertools
import math
import multiprocessing
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

# third party libraries
import fast_simplification
import numpy as np
import numpy.typing as npt
import pyvista as pv
import typer

NDArrayF64 = npt.NDArray[np.float64]


class ChunkedContour:
    """
    A class for contouring a scalar field in chunks.

    This class is useful for contouring large scalar fields that do not fit in memory.

    TODO: By surrounding the scalar field with negative infinity, we can close any holes in the
    manifold at the edges of the sampled volume.
    """

    def __init__(
        self,
        grid_x: NDArrayF64,
        grid_y: NDArrayF64,
        grid_z: NDArrayF64,
        scalar_field: Callable[[NDArrayF64], NDArrayF64],
        *,
        max_chunk_points: int,
        n_processes: int,
    ):
        """
        Initialize the ChunkedContour object.

        Args:
            grid_x: The x values on which to sample the scalar field.
            grid_y: The y values on which to sample the scalar field.
            grid_z: The z values on which to sample the scalar field.
            scalar_field: A function that takes a matrix of points and returns a vector of values.
            max_chunk_points: Maximum number of points in one chunk.
            n_processes: Number of processes to use for contouring.
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.scalar_field = scalar_field
        self.max_chunk_points = max_chunk_points
        self.n_processes = n_processes

    def _contour_chunk(self, x_slice: slice, y_slice: slice, z_slice: slice) -> Optional[pv.PolyData]:
        """Contour a chunk of the scalar field."""
        grid = pv.RectilinearGrid(self.grid_x[x_slice], self.grid_y[y_slice], self.grid_z[z_slice])
        num_points = grid.GetNumberOfPoints()
        if num_points > self.max_chunk_points:
            print(f"Chunk has too many points: {num_points}")
            return None
        points = grid.points
        print(f"contouring chunk: {x_slice=}, {y_slice=}, {z_slice=}, {points.shape=}")
        values = self.scalar_field(points)
        return grid.contour([0], values)

    def _split_array(self, length: int, chunks: int) -> Sequence[slice]:
        """Generate overlapping slices for an array."""
        n_cells = length - 1
        if not (1 <= chunks <= n_cells):
            raise ValueError(f"Invalid number of chunks given length: {chunks=}, {length=}")
        small_chunk_size, n_large_chunks = divmod(n_cells, chunks)
        large_chunk_size = small_chunk_size + 1
        slices: List[slice] = []
        low = 0
        for chunk_index in range(chunks):
            if chunk_index < n_large_chunks:
                high = low + large_chunk_size
            else:
                high = low + small_chunk_size
            slices.append(slice(low, high + 1))
            low = high
        return slices

    def _merge_meshes(self, meshes: List[pv.PolyData]) -> pv.PolyData:
        """Merge meshes."""
        return pv.PolyData().merge(meshes, progress_bar=True)

    def _contour_by_chunking(self, n_x_chunks: int, n_y_chunks: int, n_z_chunks: int) -> Optional[pv.PolyData]:
        """Contour multiple chunks and connect the results."""
        x_slices = self._split_array(len(self.grid_x), n_x_chunks)
        y_slices = self._split_array(len(self.grid_y), n_y_chunks)
        z_slices = self._split_array(len(self.grid_z), n_z_chunks)

        chunk_meshes: List[pv.PolyData] = []
        for x_slice, y_slice, z_slice in itertools.product(x_slices, y_slices, z_slices):
            chunk_mesh = self._contour_chunk(x_slice, y_slice, z_slice)
            if chunk_mesh is None:
                return None
            chunk_meshes.append(chunk_mesh)
        return self._merge_meshes(chunk_meshes)

    def _contour_by_chunking_with_multiprocessing(
        self, n_x_chunks: int, n_y_chunks: int, n_z_chunks: int
    ) -> Optional[pv.PolyData]:
        """Contour multiple chunks and connect the results using multiprocessing."""
        x_slices = self._split_array(len(self.grid_x), n_x_chunks)
        y_slices = self._split_array(len(self.grid_y), n_y_chunks)
        z_slices = self._split_array(len(self.grid_z), n_z_chunks)

        with multiprocessing.Pool(self.n_processes) as pool:
            chunk_meshes = pool.starmap(self._contour_chunk, itertools.product(x_slices, y_slices, z_slices))
        nn_chunk_meshes: List[pv.PolyData] = []
        for chunk_mesh in chunk_meshes:
            if chunk_mesh is None:
                return None
            nn_chunk_meshes.append(chunk_mesh)
        return self._merge_meshes(nn_chunk_meshes)

    def _generate_chunking_strategies(self) -> Iterator[Tuple[int, int, int]]:
        """Generate chunking strategies."""
        strategy = [1, 1, 1]
        max_chunks = (len(self.grid_x) - 1, len(self.grid_y) - 1, len(self.grid_z) - 1)

        if any(s > mc for s, mc in zip(strategy, max_chunks)):
            return
        yield tuple(strategy)  # type: ignore

        # build a linked-list-like data structure, next_index, to iterate through strategy indices
        strategy_indices = np.argsort(max_chunks)[::-1]
        next_index = [0] * len(strategy_indices)
        for strategy_index_index, strategy_index in enumerate(strategy_indices):
            next_index[strategy_indices[(strategy_index_index - 1) % len(strategy_indices)]] = strategy_index
        current_strategy_index = strategy_indices[0]
        previous_strategy_index = strategy_indices[-1]
        del strategy_indices

        while True:
            if strategy[current_strategy_index] < max_chunks[current_strategy_index]:
                current_value = strategy[current_strategy_index]
                max_value = max_chunks[current_strategy_index]
                strategy[current_strategy_index] = min(max_value, current_value * 2)

                yield tuple(strategy)  # type: ignore

                previous_strategy_index = current_strategy_index
                current_strategy_index = next_index[current_strategy_index]

            else:  # strategy[current_strategy_index] == max_chunks[current_strategy_index]
                if next_index[current_strategy_index] == current_strategy_index:
                    break

                current_strategy_index = next_index[current_strategy_index]
                next_index[previous_strategy_index] = current_strategy_index

    def contour(self) -> pv.PolyData:
        """Contour the scalar field."""
        for chunking_strategy in self._generate_chunking_strategies():
            print(f"Trying chunking strategy: {chunking_strategy}")
            if self.n_processes == 1:
                mesh = self._contour_by_chunking(*chunking_strategy)
            else:
                mesh = self._contour_by_chunking_with_multiprocessing(*chunking_strategy)
            if mesh is not None:
                return mesh
        raise RuntimeError("Could not find a chunking strategy that fits in memory.")


def vase_scalar_field(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculate values in a scalar field whose isosurface at zero defines the wall of a vase."""
    x, y, xy, z = points[:, 0], points[:, 1], points[:, :2], points[:, 2]
    theta = np.arctan2(y, x)
    r = np.linalg.norm(xy, axis=1)
    z_stretch = (z - 1) * 1.25
    r_of_z = 2 * np.sin(-0.07 * z_stretch**2 + 0.9 * z_stretch + np.pi / 4 - 0.2) + 2.5
    amplitude_factor_of_z = r_of_z / 5
    return -r + (r_of_z + 0.3 * amplitude_factor_of_z * np.sin(50 * theta + 15 * np.sin(z) + 10 * z)) * (
        (z >= 1) & (z <= 9)
    )


def _linspace_axis(resolution: float, length: float, center: bool) -> NDArrayF64:
    start = (-length / 2) if center else 0
    stop = start + length
    num = int(math.ceil(length / resolution)) + 1
    return np.linspace(start, stop, num)


def build_grid_axes(
    build_volume: Tuple[float, float, float],
    xy_resolution: float,
    z_resolution: float,
    extra_resolution_factor: float,
) -> Tuple[NDArrayF64, NDArrayF64, NDArrayF64]:
    """Build a grid from 3D printer specs."""
    xy_resolution /= extra_resolution_factor
    z_resolution /= extra_resolution_factor

    return (
        _linspace_axis(xy_resolution, build_volume[0], center=True),
        _linspace_axis(xy_resolution, build_volume[1], center=True),
        _linspace_axis(z_resolution, build_volume[2], center=False),
    )


def main(out_file: str = "vase.stl", draft: bool = True) -> None:
    """Create a mesh for a printable vase and save as an stl."""
    extra_resolution_factor = 1.0 if draft else 10.0

    grid_x, grid_y, grid_z = build_grid_axes(
        build_volume=(10.0, 10.0, 10.0),
        xy_resolution=0.05,
        z_resolution=0.05,
        extra_resolution_factor=extra_resolution_factor,
    )

    cc = ChunkedContour(
        grid_x, grid_y, grid_z, vase_scalar_field, max_chunk_points=50_000_000, n_processes=multiprocessing.cpu_count()
    )
    mesh = cc.contour()

    if not draft:
        print("Simplifying mesh now.")
        simplified_mesh = fast_simplification.simplify_mesh(mesh, target_reduction=0.9, verbose=True)
        print("Done simplifying the mesh.")
        simplified_mesh.save(out_file)
    else:
        mesh.save(out_file)


if __name__ == "__main__":
    typer.run(main)
