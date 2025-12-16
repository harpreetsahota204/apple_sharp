import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement

import fiftyone as fo
from fiftyone import Model
from fiftyone.core.models import SupportsGetItem, TorchModelMixin
from fiftyone.utils.torch import GetItem

# SHARP model imports (install via: pip install git+https://github.com/apple/ml-sharp#egg=sharp)
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import save_ply
from sharp.utils.io import load_rgb
from sharp.cli.predict import predict_image

logger = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
INTERNAL_SHAPE = (1536, 1536)
SH_C0 = 0.28209479177387814


@dataclass
class SHARPConfig:
    """Configuration for SHARP FiftyOne model."""
    model_path: Optional[str] = None
    device: str = "auto"
    output_dir: Optional[str] = None
    save_3dgs_ply: bool = True
    default_focal_length_mm: float = 30.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def convert_3dgs_ply(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """Convert 3DGS PLY to standard PLY with RGB colors for FiftyOne viewer.
    
    Preserves all original 3DGS properties for round-trip compatibility.
    """
    input_path = Path(input_path)
    output_path = output_path or input_path.with_stem(f"{input_path.stem}_rgb")

    ply = PlyData.read(input_path)
    vertices = ply["vertex"]
    props = [p.name for p in vertices.properties]
    n_points = len(vertices["x"])

    # Build output dtype - position and RGB first
    dtype_fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    dtype_fields.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])

    # Alpha if opacity exists
    has_opacity = "opacity" in props
    if has_opacity:
        dtype_fields.append(("alpha", "u1"))

    # Scale as normals (nx, ny, nz)
    has_scale = all(f"scale_{i}" in props for i in range(3))
    if has_scale:
        dtype_fields.extend([("nx", "f4"), ("ny", "f4"), ("nz", "f4")])

    # Rotation quaternion
    has_rotation = all(f"rot_{i}" in props for i in range(4))
    if has_rotation:
        dtype_fields.extend([("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")])

    # Higher order SH (f_rest_*)
    sh_rest_props = sorted([p for p in props if p.startswith("f_rest_")])
    for prop in sh_rest_props:
        dtype_fields.append((prop, "f4"))

    # Original f_dc_* for round-trip
    dtype_fields.extend([("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")])

    # Original opacity and scale for round-trip
    if has_opacity:
        dtype_fields.append(("opacity_raw", "f4"))
    if has_scale:
        dtype_fields.extend([("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")])

    # Create output array
    out = np.zeros(n_points, dtype=dtype_fields)

    # Position
    out["x"] = vertices["x"]
    out["y"] = vertices["y"]
    out["z"] = vertices["z"]

    # Convert SH DC to RGB
    f_dc_0 = np.asarray(vertices["f_dc_0"]) if "f_dc_0" in props else np.zeros(n_points)
    f_dc_1 = np.asarray(vertices["f_dc_1"]) if "f_dc_1" in props else np.zeros(n_points)
    f_dc_2 = np.asarray(vertices["f_dc_2"]) if "f_dc_2" in props else np.zeros(n_points)

    out["red"] = np.clip((0.5 + SH_C0 * f_dc_0) * 255, 0, 255).astype(np.uint8)
    out["green"] = np.clip((0.5 + SH_C0 * f_dc_1) * 255, 0, 255).astype(np.uint8)
    out["blue"] = np.clip((0.5 + SH_C0 * f_dc_2) * 255, 0, 255).astype(np.uint8)

    # Store original SH DC
    out["f_dc_0"] = f_dc_0
    out["f_dc_1"] = f_dc_1
    out["f_dc_2"] = f_dc_2

    # Opacity -> Alpha
    if has_opacity:
        opacity_raw = np.asarray(vertices["opacity"])
        out["alpha"] = np.clip(sigmoid(opacity_raw) * 255, 0, 255).astype(np.uint8)
        out["opacity_raw"] = opacity_raw

    # Scale (stored as log, convert to exp for normals)
    if has_scale:
        scale_0 = np.asarray(vertices["scale_0"])
        scale_1 = np.asarray(vertices["scale_1"])
        scale_2 = np.asarray(vertices["scale_2"])
        out["nx"] = np.exp(scale_0)
        out["ny"] = np.exp(scale_1)
        out["nz"] = np.exp(scale_2)
        out["scale_0"] = scale_0
        out["scale_1"] = scale_1
        out["scale_2"] = scale_2

    # Rotation quaternion
    if has_rotation:
        out["rot_0"] = vertices["rot_0"]
        out["rot_1"] = vertices["rot_1"]
        out["rot_2"] = vertices["rot_2"]
        out["rot_3"] = vertices["rot_3"]

    # Higher order SH
    for prop in sh_rest_props:
        out[prop] = vertices[prop]

    # Write output
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(str(output_path))
    return output_path


class SHARPGetItem(GetItem):
    """Data loader transform for SHARP model. Just passes filepath to predict_all."""

    def __init__(self, field_mapping: Optional[Dict[str, str]] = None):
        super().__init__(field_mapping=field_mapping)

    @property
    def required_keys(self) -> List[str]:
        return ["filepath"]

    def __call__(self, sample_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Just return filepath - SHARP's load_rgb handles the rest."""
        return {"filepath": sample_dict["filepath"]}


class SHARPModel(Model, SupportsGetItem, TorchModelMixin):
    """SHARP model wrapper for 3D Gaussian Splat prediction from single images."""

    def __init__(self, config: Optional[SHARPConfig] = None):
        SupportsGetItem.__init__(self)
        self._preprocess = False
        self.config = config or SHARPConfig()
        self._device = self._get_device(self.config.device)
        self._model = self._load_model()

    def _get_device(self, device_str: str) -> torch.device:
        if device_str != "auto":
            return torch.device(device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "mps") and torch.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self) -> torch.nn.Module:
        logger.info("Loading SHARP model...")
        predictor = create_predictor(PredictorParams())

        if self.config.model_path:
            state_dict = torch.load(self.config.model_path, map_location=self._device, weights_only=True)
        else:
            state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, map_location=self._device, progress=True)

        predictor.load_state_dict(state_dict)
        predictor.eval()
        predictor.to(self._device)
        logger.info(f"SHARP model loaded on {self._device}")
        return predictor

    # Properties from Model base class
    @property
    def media_type(self) -> str:
        return "image"

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self) -> bool:
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value: bool):
        self._preprocess = value

    @property
    def ragged_batches(self) -> bool:
        return False

    # Properties from TorchModelMixin
    @property
    def has_collate_fn(self) -> bool:
        return True

    @property
    def collate_fn(self):
        return lambda batch: batch

    # Methods from SupportsGetItem
    def build_get_item(self, field_mapping: Optional[Dict[str, str]] = None) -> GetItem:
        return SHARPGetItem(field_mapping=field_mapping)

    def predict(self, arg) -> Optional[str]:
        """Process a single sample. Expects dict with 'filepath' key."""
        if isinstance(arg, dict):
            results = self.predict_all([arg])
            return results[0] if results else None
        # If given a filepath string directly
        if isinstance(arg, (str, Path)):
            results = self.predict_all([{"filepath": str(arg)}])
            return results[0] if results else None
        raise ValueError("predict() expects a dict with 'filepath' or a filepath string")

    def predict_all(self, batch: List[Optional[Dict[str, Any]]]) -> List[Optional[str]]:
        """Process a batch of images using SHARP's predict_image."""
        results = []

        for item in batch:
            if item is None:
                results.append(None)
                continue

            filepath = item["filepath"]
            
            # Use SHARP's load_rgb to get image and focal length (same as CLI)
            image, _, f_px = load_rgb(Path(filepath))
            height, width = image.shape[:2]

            # Use SHARP's predict_image (same as CLI)
            gaussians = predict_image(self._model, image, f_px, self._device)

            # Output paths
            source_path = Path(filepath)
            output_dir = Path(self.config.output_dir) if self.config.output_dir else source_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            stem = source_path.stem
            ply_path = output_dir / f"{stem}_sharp.ply"
            rgb_ply_path = output_dir / f"{stem}_sharp_rgb.ply"
            fo3d_path = output_dir / f"{stem}_sharp.fo3d"

            # Save PLY and convert
            save_ply(gaussians, f_px, (height, width), ply_path)
            convert_3dgs_ply(ply_path, rgb_ply_path)

            if not self.config.save_3dgs_ply:
                ply_path.unlink()

            # Create fo3d scene
            scene = fo.Scene()
            scene.camera = fo.PerspectiveCamera(up="Z")
            mesh = fo.PlyMesh("mesh", str(rgb_ply_path), is_point_cloud=True)
            scene.add(mesh)
            scene.write(str(fo3d_path))

            results.append(str(fo3d_path))

        return results
