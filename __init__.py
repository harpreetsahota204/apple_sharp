import os
import logging
import urllib.request

from fiftyone.operators import types

from sharp.zoo import SHARPModel, SHARPConfig

logger = logging.getLogger(__name__)

# Model download URL
SHARP_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

# Supported model variants
MODEL_VARIANTS = {
    "Apple/SHARP": {
        "url": SHARP_MODEL_URL,
        "filename": "sharp_2572gikvuh.pt",
    },
}


def download_model(model_name, model_path):
    """Downloads the SHARP model weights.
    
    Args:
        model_name: the name of the model to download
        model_path: the absolute filename or directory to which to download the model
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported model name '{model_name}'. "
            f"Supported models: {list(MODEL_VARIANTS.keys())}"
        )
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    url = MODEL_VARIANTS[model_name]["url"]
    
    logger.info(f"Downloading SHARP model from {url}...")
    
    # Download the model weights
    urllib.request.urlretrieve(url, model_path)
    
    logger.info(f"SHARP model saved to {model_path}")


def load_model(
    model_name, 
    model_path, 
    device="auto",
    output_dir=None,
    save_3dgs_ply=True,
    default_focal_length_mm=30.0,
    **kwargs
):
    """Loads the SHARP model.
    
    Args:
        model_name: the name of the model to load
        model_path: the absolute filename to which the model was downloaded
        device: device to run inference on ("auto", "cuda", "cpu", "mps")
        output_dir: directory for output files (None = alongside source images)
        save_3dgs_ply: whether to save original 3DGS PLY format
        default_focal_length_mm: fallback focal length if EXIF missing
        **kwargs: additional keyword arguments
        
    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported model name '{model_name}'. "
            f"Supported models: {list(MODEL_VARIANTS.keys())}"
        )

    config = SHARPConfig(
        model_path=model_path,
        device=device,
        output_dir=output_dir,
        save_3dgs_ply=save_3dgs_ply,
        default_focal_length_mm=default_focal_length_mm,
    )
    
    return SHARPModel(config)


def resolve_input(model_name, ctx):
    """Defines any necessary properties to collect the model's custom
    parameters from a user during prompting.
    
    Args:
        model_name: the name of the model
        ctx: an :class:`fiftyone.operators.ExecutionContext`
        
    Returns:
        a :class:`fiftyone.operators.types.Property`, or None
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported model name '{model_name}'. "
            f"Supported models: {list(MODEL_VARIANTS.keys())}"
        )
    
    inputs = types.Object()
    
    inputs.enum(
        "device",
        values=["auto", "cuda", "cpu", "mps"],
        default="auto",
        label="Device",
        description="Device to run inference on"
    )
    
    inputs.bool(
        "save_3dgs_ply",
        default=True,
        label="Save 3DGS PLY",
        description="Whether to save original 3DGS PLY format (for use with other renderers)"
    )
    
    inputs.float(
        "default_focal_length_mm",
        default=30.0,
        label="Default Focal Length (mm)",
        description="Fallback focal length (35mm equivalent) if EXIF data is missing"
    )
    
    return types.Property(inputs)
