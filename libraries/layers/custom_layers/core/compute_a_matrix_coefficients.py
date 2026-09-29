import logging
import numpy as np
from samples.layers.core.base import Layer, SampleT

logger = logging.getLogger(__name__)


class ComputeAMatrixCoefficients(Layer):
    """Compute A-matrix coefficients and error estimates from Q and T sub-matrices.

    Reads Q01–Q{n} and T01–T{n} arrays from a scalar_data datablock (typically
    imported from a heatFluxMatrix CSV), then writes back:

    - ``q_matrix``                      : stacked Q arrays, shape (n, N)
    - ``q_error``                       : column-wise sum of q_matrix, shape (N,); should be ≤ 1
    - ``t_matrix``                      : stacked T arrays, shape (n, N)
    - ``a_matrix``                      : q_matrix @ inv(t_matrix), shape (n, n)
    - ``a_matrix_coefficients``         : upper-triangular off-diagonal entries of a_matrix
    - ``a_matrix_error``                : scalar symmetry error sqrt(sum((A - A^T)^2))
    - "geometry_parameter_a_matrix"     : 120*20 that repeats stator_scalars geometry values of 20 parameters 120 times and stacks over to form the matrix.
    - "a_matrix_coeffiecient positions" : position of a_matrx values in heatmap (16*16) before we converted it into trie_index matric. 

    Args:
        input_datablock: Name of the scalar_data datablock containing the Q/T arrays.
        n_components: Number of Q/T components (columns labelled Q01…Q{n} and T01…T{n}).
        tags: Optional set of tags for conditional execution.
    """

    def __init__(
        self,
        input_datablock: str,
        *,
        n_components: int = 16,
        tags: set[str] | None = None,
    ) -> None:
        super().__init__(tags=tags)
        if not input_datablock:
            raise ValueError("input_datablock must not be empty")
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.input_datablock = input_datablock
        self.n_components = n_components

    def geometry_constraints(self, subset,LOWER_BOUND, UPPER_BOUND) -> bool:
     
        # Check if any element violates the constraint
        return any(x < LOWER_BOUND or x > UPPER_BOUND for x in subset)

    def __call__(self, sample: SampleT) -> SampleT:
        datablock = sample.get(self.input_datablock)
        # datablock is loaded from the H5 file
        if datablock is None:
            logger.warning(
                "Datablock '%s' not found in sample '%s'; "
                "skipping A-matrix computation.",
                self.input_datablock,
                sample.uuid,
            )
            flag_data = {
                "missing_heatflux": np.array([True]),
                "high_q_error": np.array([False]),
                "high_a_matrix_error": np.array([False]),
            }
            flags = sample.get("flags")
            if flags is None:
                sample.add("flags", flag_data, "scalar_data")
            else:
                for name, value in flag_data.items():
                    flags.write_array(name, value)
            return sample

        # Stack Q and T sub-matrices from named arrays
        q_keys = [f"Q{str(i).zfill(2)}" for i in range(1, self.n_components + 1)]
        t_keys = [f"T{str(i).zfill(2)}" for i in range(1, self.n_components + 1)]

        q_matrix = np.stack([datablock.read_array(k) for k in q_keys])  # (n, N)
        t_matrix = np.stack([datablock.read_array(k) for k in t_keys])  # (n, N)

        # Q error: column-wise row sum; a well-conditioned matrix has values ≤ 1
        q_error = np.sqrt(q_matrix.sum(axis=0)**2)  # (N,)
        high_q_error = bool(np.any(q_error > 1))
        if high_q_error:
            logger.warning(
                "Sample '%s' has q_error > 1 (max=%.4f).",
                sample.uuid,
                float(q_error.max()),
            )

        # A matrix: q_matrix @ inv(t_matrix)  →  (n, n)
        a_matrix = q_matrix @ np.linalg.inv(t_matrix)

        # Symmetry error of A
        a_matrix_error = float(np.sqrt(np.sum((a_matrix - a_matrix.T) ** 2)))

        # Upper-triangular off-diagonal coefficients
        a_matrix_coefficients = a_matrix[np.triu_indices_from(a_matrix, k=1)]

        # geometry_parameter_a_matrix
        geom_dict = sample.get("stator_scalars").to_dict() # dict 
        geom_param_names = sorted(list(geom_dict.keys()))
        geom_array = np.zeros(len(geom_param_names))
        for i, geom_param_name in enumerate(geom_param_names):
            geom_array[i] = geom_dict[geom_param_name]
        geometry_parameter_a_matrix = np.tile(geom_array, (120, 1))  # (120, 20)
        
        # a_matrix_coeffiecient positions
        rows, cols = np.triu_indices_from(a_matrix, k=1)
        a_matrix_coeffiecient_positions = np.stack([rows, cols], axis=1)   # shape (n_coeff, 2)

        datablock.write_array("q_matrix", q_matrix)
        datablock.write_array("q_error", q_error)
        datablock.write_array("t_matrix", t_matrix)
        datablock.write_array("a_matrix", a_matrix)
        datablock.write_array("a_matrix_error", np.array([a_matrix_error]))
        datablock.write_array("a_matrix_coefficients", a_matrix_coefficients)
        datablock.write_array("geometry_parameter_a_matrix", geometry_parameter_a_matrix)
        datablock.write_array("a_matrix_coeffiecient_positions",  a_matrix_coeffiecient_positions)
        
        
        # A matrix error:
        high_a_matrix_error = a_matrix_error > 1
        if high_a_matrix_error:
            logger.warning(
                "Sample '%s' has a_matrix_error > 1 (value=%.4f).",
                sample.uuid,
                a_matrix_error,
            )

        flag_data = {
            "missing_heatflux": np.array([False]),
            "high_q_error": np.array([high_q_error]),
            "high_a_matrix_error": np.array([high_a_matrix_error]),
            "geometry_constraints_violated": np.array([self.geometry_constraints(geom_array[14:18], 0.3, 1.835)])
        }
        flags = sample.get("flags")
        if flags is None:
            sample.add("flags", flag_data, "scalar_data")
        else:
            for name, value in flag_data.items():
                flags.write_array(name, value)

        return sample
