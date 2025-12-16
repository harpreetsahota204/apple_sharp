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
import torch.nn.functional as F
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians
from sharp.utils.io import load_rgb

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


def convert_3dgs_ply(input_path: Path, output_path: Path = None) -> Path:
    """Convert 3DGS PLY to standard PLY with RGB colors (proven fast approach)."""
    input_path = Path(input_path)
    output_path = output_path or input_path.with_stem(f"{input_path.stem}_rgb")

    ply = PlyData.read(input_path)
    vertices = ply["vertex"]
    props = [p.name for p in vertices.properties]
    n_points = len(vertices["x"])

    # Build dtype - position and RGB first
    dtype_fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    dtype_fields.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])

    has_opacity = "opacity" in props
    if has_opacity:
        dtype_fields.append(("alpha", "u1"))

    has_scale = all(f"scale_{i}" in props for i in range(3))
    if has_scale:
        dtype_fields.extend([("nx", "f4"), ("ny", "f4"), ("nz", "f4")])

    has_rotation = all(f"rot_{i}" in props for i in range(4))
    if has_rotation:
        dtype_fields.extend([("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")])

    sh_rest_props = sorted([p for p in props if p.startswith("f_rest_")])
    for prop in sh_rest_props:
        dtype_fields.append((prop, "f4"))

    dtype_fields.extend([("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")])

    if has_opacity:
        dtype_fields.append(("opacity_raw", "f4"))
    if has_scale:
        dtype_fields.extend([("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")])

    out = np.zeros(n_points, dtype=dtype_fields)

    # Position
    out["x"], out["y"], out["z"] = vertices["x"], vertices["y"], vertices["z"]

    # SH DC to RGB
    f_dc_0 = vertices["f_dc_0"] if "f_dc_0" in props else np.zeros(n_points)
    f_dc_1 = vertices["f_dc_1"] if "f_dc_1" in props else np.zeros(n_points)
    f_dc_2 = vertices["f_dc_2"] if "f_dc_2" in props else np.zeros(n_points)

    out["red"] = np.clip((0.5 + SH_C0 * f_dc_0) * 255, 0, 255).astype(np.uint8)
    out["green"] = np.clip((0.5 + SH_C0 * f_dc_1) * 255, 0, 255).astype(np.uint8)
    out["blue"] = np.clip((0.5 + SH_C0 * f_dc_2) * 255, 0, 255).astype(np.uint8)
    out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = f_dc_0, f_dc_1, f_dc_2

    if has_opacity:
        opacity_raw = vertices["opacity"]
        out["alpha"] = np.clip(1 / (1 + np.exp(-opacity_raw)) * 255, 0, 255).astype(np.uint8)
        out["opacity_raw"] = opacity_raw

    if has_scale:
        out["nx"], out["ny"], out["nz"] = np.exp(vertices["scale_0"]), np.exp(vertices["scale_1"]), np.exp(vertices["scale_2"])
        out["scale_0"], out["scale_1"], out["scale_2"] = vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]

    if has_rotation:
        out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]

    for prop in sh_rest_props:
        out[prop] = vertices[prop]

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

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit - clear GPU memory cache."""
        # Clear cache based on device type (don't move model to CPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False

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
        """Process a batch of images with batched forward pass."""
        # Track valid items
        valid_indices = []
        images_tensors = []
        disparity_factors = []
        metadata = []

        for i, item in enumerate(batch):
            if item is None:
                continue

            filepath = item["filepath"]
            
            # Use SHARP's load_rgb (same as CLI)
            image, _, f_px = load_rgb(Path(filepath))
            height, width = image.shape[:2]

            # Preprocess (same as predict_image)
            image_pt = torch.from_numpy(image.copy()).float().to(self._device).permute(2, 0, 1) / 255.0
            image_resized = F.interpolate(image_pt.unsqueeze(0), size=INTERNAL_SHAPE, mode="bilinear", align_corners=True).squeeze(0)

            valid_indices.append(i)
            images_tensors.append(image_resized)
            disparity_factors.append(f_px / width)
            metadata.append({"filepath": filepath, "width": width, "height": height, "f_px": f_px})

        # Initialize results
        results = [None] * len(batch)
        
        if not valid_indices:
            return results

        # Batched forward pass
        images_batch = torch.stack(images_tensors)
        disparity_batch = torch.tensor(disparity_factors, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            gaussians_ndc = self._model(images_batch, disparity_batch)

        # Post-process each result using SHARP's unproject_gaussians
        for idx, (orig_idx, meta) in enumerate(zip(valid_indices, metadata)):
            filepath = meta["filepath"]
            width, height, f_px = meta["width"], meta["height"], meta["f_px"]

            # Extract single result from batch
            gaussians_i = Gaussians3D(
                mean_vectors=gaussians_ndc.mean_vectors[idx:idx+1],
                singular_values=gaussians_ndc.singular_values[idx:idx+1],
                quaternions=gaussians_ndc.quaternions[idx:idx+1],
                colors=gaussians_ndc.colors[idx:idx+1],
                opacities=gaussians_ndc.opacities[idx:idx+1],
            )

            # Compute intrinsics (same as predict_image)
            intrinsics = torch.tensor([
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32, device=self._device)
            intrinsics[0] *= INTERNAL_SHAPE[0] / width
            intrinsics[1] *= INTERNAL_SHAPE[1] / height

            # Unproject using SHARP's function (same as predict_image)
            gaussians = unproject_gaussians(
                gaussians_i, 
                torch.eye(4, device=self._device), 
                intrinsics, 
                INTERNAL_SHAPE
            )

            # Output paths (use absolute paths for fo3d compatibility)
            source_path = Path(filepath).resolve()
            output_dir = Path(self.config.output_dir).resolve() if self.config.output_dir else source_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            stem = source_path.stem
            ply_path = output_dir / f"{stem}_sharp.ply"
            fo3d_path = output_dir / f"{stem}_sharp.fo3d"

            # Use SHARP's save_ply, then convert in-place (saves disk space)
            save_ply(gaussians, f_px, (height, width), ply_path)
            convert_3dgs_ply(ply_path, ply_path)  # Overwrites with RGB version

            # Create fo3d scene
            scene = fo.Scene()
            scene.camera = fo.PerspectiveCamera(up="Z")
            mesh = fo.PlyMesh("mesh", str(ply_path), is_point_cloud=True)
            scene.add(mesh)
            scene.write(str(fo3d_path))

            results[orig_idx] = str(fo3d_path)

        return results
