# Apple SHARP - FiftyOne Model Zoo Integration

![image](sharp.gif)

[SHARP](https://github.com/apple/ml-sharp) is Apple's state-of-the-art model for predicting 3D Gaussian Splats from a single RGB image. This integration brings SHARP to FiftyOne, enabling batch inference on image datasets with 3D visualization.


## Installation

```bash
pip install sharp@git+https://github.com/apple/ml-sharp
pip install fiftyone
```

## Quick Start

```python
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.huggingface import load_from_hub

# Register the model source
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/apple_sharp",
    overwrite=True
)

dataset = load_from_hub(
    "Voxel51/sku110k_test",
    max_samples=5,
    overwrite=True
    )


# Load and apply the model
model = foz.load_zoo_model("Apple/SHARP")

dataset.apply_model(
    model,
    "threed_path",
    batch_size=16,
    num_workers=4
)
```

## Creating a Grouped Dataset for Multi-Modal Visualization

The model runs and saves results (`.ply` and `.fo3d` files) alongside the original input files, but these are not automatically added to your dataset. Use the following code to create a grouped dataset for comprehensive visualization of both RGB images and 3D point clouds:

```python
import fiftyone as fo
import os
from pathlib import Path

# Get filepaths from your existing dataset
filepaths = dataset.values("filepath")

# Create a new grouped dataset
grouped_dataset = fo.Dataset("sharp_results", overwrite=True)
grouped_dataset.add_group_field("group", default="rgb")

# Process each filepath and create the group structure
samples = []
for filepath in filepaths:
    path = Path(filepath)
    base_dir = path.parent
    base_name = path.stem

    # Create paths for each modality
    rgb_path = filepath  # Original RGB image
    threed_path = os.path.join(base_dir, f"{base_name}.fo3d")  # 3D scene

    # Create a group for these related samples
    group = fo.Group()

    # Create samples for each modality
    rgb_sample = fo.Sample(filepath=rgb_path, group=group.element("rgb"))
    threed_sample = fo.Sample(filepath=threed_path, group=group.element("threed"))

    samples.extend([rgb_sample, threed_sample])

# Add all samples to the dataset
grouped_dataset.add_samples(samples)

# Launch the app
session = fo.launch_app(grouped_dataset)
```

## Rendering Colors in FiftyOne App

To render the point cloud colors properly in the FiftyOne App:

1. Open the 3D scene view
2. Click on **Render Preferences**
3. Set **"Shade by"** to **None**

This ensures the RGB colors from the PLY file are displayed directly rather than being overridden by shading.

## Performance Notes

While the SHARP neural network itself runs inference very fast (~1 second per image on GPU), the overall pipeline includes additional operations that add some overhead:

1. **Image Loading**: SHARP's `load_rgb` reads EXIF data for focal length extraction
2. **3D Unprojection**: Converting NDC Gaussians to metric space involves CPU-based SVD decomposition (~200k points per image)
3. **PLY I/O**: Writing the 3DGS PLY file and converting it to standard PLY format
4. **fo3d Scene Creation**: Writing the FiftyOne scene file

These read/write operations are necessary for compatibility with standard PLY viewers and FiftyOne's 3D visualization.

---

## Technical Details: Converting 3DGS PLY to Standard PLY

Apple's SHARP outputs 3D Gaussian Splat (3DGS) files in PLY format with specialized properties that standard PLY viewers don't understand. This integration converts them to viewer-compatible PLY files with standard RGB colors.

### Color Encoding (Spherical Harmonics)

3DGS stores color using **spherical harmonics (SH)** rather than direct RGB values. The `f_dc_0`, `f_dc_1`, `f_dc_2` properties represent the DC (zero-order) SH coefficients for each color channel.

To convert SH to RGB:

```
RGB = 0.5 + SH_C0 × f_dc
```

Where `SH_C0 = 0.28209...` is the zero-order SH basis function constant (`1 / (2√π)`). The 0.5 offset centers the color range since SH coefficients are centered around zero.

### Opacity Encoding (Logit Space)

3DGS stores opacity in **logit space** (log-odds) to allow unconstrained optimization during training. The raw values range from negative to positive infinity. We apply the **sigmoid function** to map them back to [0, 1]:

```
alpha = sigmoid(opacity) = 1 / (1 + e^(-opacity))
```

### Scale Encoding (Log Space)

Scale values are stored in **log space** to ensure they remain positive during optimization. We apply `exp()` to recover the actual scale:

```
scale_actual = exp(scale_raw)
```

### Preserved Properties

The converted PLY files preserve all original 3DGS properties for round-trip compatibility:

| Property | Description |
|----------|-------------|
| `x, y, z` | Point positions |
| `red, green, blue` | Converted RGB colors (0-255) |
| `alpha` | Converted opacity (0-255) |
| `nx, ny, nz` | Linear scale values (exp of log-scale) |
| `rot_0-3` | Rotation quaternion |
| `f_dc_0-2` | Original SH DC coefficients |
| `f_rest_*` | Higher-order SH coefficients |
| `opacity_raw` | Original logit opacity |
| `scale_0-2` | Original log-scale values |

---

## License

This integration is provided under the same license as the original [ml-sharp](https://github.com/apple/ml-sharp) repository. See their LICENSE file for details.

## Acknowledgements

- [Apple ML Research](https://github.com/apple) for the SHARP model
- [FiftyOne](https://github.com/voxel51/fiftyone) for the dataset and model zoo infrastructure

# Citation

```bibtex
@inproceedings{Sharp2025:arxiv,
  title      = {Sharp Monocular View Synthesis in Less Than a Second},
  author     = {Lars Mescheder and Wei Dong and Shiwei Li and Xuyang Bai and Marcel Santos and Peiyun Hu and Bruno Lecouat and Mingmin Zhen and Ama\"{e}l Delaunoyand Tian Fang and Yanghai Tsin and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2512.10685},
  year       = {2025},
  url        = {https://arxiv.org/abs/2512.10685},
}
```