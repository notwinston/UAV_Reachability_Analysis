"""Value function loader for the defender controller.

Loads precomputed value functions from .npz files and provides
interpolation and gradient computation for real-time control.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

# Try to import from reach_avoid_game; fall back to direct numpy loading
try:
    from reach_avoid_game.solvers.value_function_io import load_value_function, ValueFunctionData
    _HAS_RAG = True
except ImportError:
    _HAS_RAG = False

    class ValueFunctionData:
        """Minimal fallback container when reach_avoid_game is not installed."""
        def __init__(self, values, grid_min, grid_max, grid_shape, params=None, description=""):
            self.values = values
            self.grid_min = grid_min
            self.grid_max = grid_max
            self.grid_shape = grid_shape
            self.params = params or {}
            self.description = description

    def load_value_function(path):
        path = Path(path)
        with np.load(path, allow_pickle=True) as npz:
            # params field is a pickled dict that may reference numpy._core
            # (saved with numpy 2.x). Gracefully handle if unpickling fails.
            try:
                params_raw = npz["params"]
                if params_raw.ndim == 0:
                    params = params_raw.item()
                else:
                    params = dict(params_raw)
            except (ModuleNotFoundError, ImportError):
                logger.warning("Could not unpickle params from %s (numpy version mismatch), using defaults", path)
                params = {}
            return ValueFunctionData(
                values=npz["values"],
                grid_min=npz["grid_min"],
                grid_max=npz["grid_max"],
                grid_shape=tuple(npz["grid_shape"]),
                params=params,
                description=str(npz.get("description", "unknown")),
            )


# Expected value function names
VF_NAMES = ["phi_z", "V_z_inf", "B_z", "phi_h", "V_h_T", "V_h_T_6d", "B_h", "phi_A_reach"]
PAPER_REQUIRED_NAMES = {"phi_z", "B_z", "phi_h", "B_h"}


class ValueFunctionLoader:
    """Loads value functions and provides interpolation/gradient queries.

    Attributes:
        vf_data: dict mapping name -> ValueFunctionData
        interpolators: dict mapping name -> RegularGridInterpolator
    """

    def __init__(self, value_function_dir: str | Path):
        self.vf_dir = Path(value_function_dir)
        self.vf_data: dict[str, ValueFunctionData] = {}
        self.interpolators: dict[str, RegularGridInterpolator] = {}
        self._grid_spacings: dict[str, np.ndarray] = {}

        self._load_all()

    def _load_all(self) -> None:
        """Load all value function .npz files from the directory."""
        for name in VF_NAMES:
            path = self.vf_dir / f"{name}.npz"
            if not path.exists():
                logger.warning("Value function file not found: %s", path)
                continue
            try:
                vf = load_value_function(path)
                if not self._artifact_is_usable(name, vf):
                    logger.warning("Ignoring invalid value function '%s': %s", name, path)
                    continue
                self.vf_data[name] = vf
                interp, spacings = self._build_interpolator(vf)
                self.interpolators[name] = interp
                self._grid_spacings[name] = spacings
                logger.info("Loaded value function '%s': shape=%s", name, vf.values.shape)
            except Exception:
                logger.exception("Failed to load value function '%s'", name)

    @staticmethod
    def _artifact_is_usable(name: str, vf: ValueFunctionData) -> bool:
        """Reject stale threshold-expanded artifacts that must be paper-valid."""
        params = vf.params if isinstance(vf.params, dict) else {}
        if name == "B_z" and float(params.get("d_z_effective", params.get("d_z", 1.0))) > float(params.get("d_z", 1.0)) + 1e-9:
            return False
        if name == "B_h" and float(params.get("d_h_effective", params.get("d_h", 3.0))) > float(params.get("d_h", 3.0)) + 1e-9:
            return False
        if name in PAPER_REQUIRED_NAMES and not bool(params.get("paper_valid", False)):
            return False
        return True

    @staticmethod
    def _build_interpolator(vf: ValueFunctionData):
        """Build a RegularGridInterpolator and compute grid spacings."""
        ndim = vf.values.ndim
        axes = []
        spacings = np.zeros(ndim)
        for i in range(ndim):
            n = vf.values.shape[i]
            lo = float(vf.grid_min[i])
            hi = float(vf.grid_max[i])
            axes.append(np.linspace(lo, hi, n))
            spacings[i] = (hi - lo) / max(n - 1, 1)

        interp = RegularGridInterpolator(
            tuple(axes),
            vf.values,
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate via nearest
        )
        return interp, spacings

    @property
    def loaded_names(self) -> list[str]:
        return list(self.vf_data.keys())

    @property
    def all_loaded(self) -> bool:
        return all(name in self.vf_data for name in VF_NAMES)

    def _clamp_state(self, name: str, state: np.ndarray) -> np.ndarray:
        """Clamp state to grid boundaries, logging a warning if out-of-bounds."""
        vf = self.vf_data[name]
        state = self._coerce_state(name, state)
        grid_min = np.asarray(vf.grid_min, dtype=float)
        grid_max = np.asarray(vf.grid_max, dtype=float)
        clamped = np.clip(state, grid_min, grid_max)
        if not np.allclose(state, clamped, atol=1e-6):
            logger.debug(
                "State out of bounds for '%s': %s clamped to %s", name, state, clamped
            )
        return clamped

    def _coerce_state(self, name: str, state: np.ndarray) -> np.ndarray:
        """Support legacy 4D relative B_h queries against 6D B_h artifacts."""
        vf = self.vf_data[name]
        if name == "B_h" and vf.values.ndim == 6 and len(state) == 4:
            x_rel, y_rel, v_x, v_y = state
            x_mid = 0.5 * (float(vf.grid_min[0]) + float(vf.grid_max[0]))
            y_mid = 0.5 * (float(vf.grid_min[1]) + float(vf.grid_max[1]))
            return np.array([
                x_mid + 0.5 * x_rel,
                y_mid + 0.5 * y_rel,
                v_x,
                v_y,
                x_mid - 0.5 * x_rel,
                y_mid - 0.5 * y_rel,
            ], dtype=float)
        return state

    def get_value(self, vf_name: str, state: np.ndarray) -> float:
        """Interpolate value function at a given state.

        Args:
            vf_name: Name of the value function (e.g. 'phi_z', 'B_z')
            state: State vector matching the VF dimensionality

        Returns:
            Interpolated scalar value
        """
        if vf_name not in self.interpolators:
            raise KeyError(f"Value function '{vf_name}' not loaded")
        state = np.asarray(state, dtype=float)
        clamped = self._clamp_state(vf_name, state)
        return float(self.interpolators[vf_name](clamped.reshape(1, -1))[0])

    def get_gradient(self, vf_name: str, state: np.ndarray) -> np.ndarray:
        """Compute gradient via central finite differences.

        Args:
            vf_name: Name of the value function
            state: State vector matching the VF dimensionality

        Returns:
            Gradient vector (same size as state)
        """
        if vf_name not in self.interpolators:
            raise KeyError(f"Value function '{vf_name}' not loaded")

        state = np.asarray(state, dtype=float)
        state = self._coerce_state(vf_name, state)
        clamped = self._clamp_state(vf_name, state)
        interp = self.interpolators[vf_name]
        spacings = self._grid_spacings[vf_name]
        ndim = len(clamped)
        grad = np.zeros(ndim)

        for i in range(ndim):
            h = spacings[i]
            s_plus = clamped.copy()
            s_minus = clamped.copy()
            s_plus[i] += h
            s_minus[i] -= h
            v_plus = float(interp(s_plus.reshape(1, -1))[0])
            v_minus = float(interp(s_minus.reshape(1, -1))[0])
            grad[i] = (v_plus - v_minus) / (2.0 * h)

        return grad

    def get_params(self, vf_name: str) -> dict:
        """Return the params dict stored with a value function."""
        if vf_name not in self.vf_data:
            raise KeyError(f"Value function '{vf_name}' not loaded")
        p = self.vf_data[vf_name].params
        return p if isinstance(p, dict) else {}
