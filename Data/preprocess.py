from __future__ import annotations
import ast
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import argparse


try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import pydicom
except Exception:
    pydicom = None

try:
    from dicom2nifti import common as d2n_common
except Exception:
    d2n_common = None


parser = argparse.ArgumentParser(description = "67")
parser.add_argument("--labels", type=str, default=None, help="The root of the labels file")
parser.add_argument("--localizers", type=str,default=None, help="The root of the localizers file")
parser.add_argument("--series", type=str,default=None, help="The root of the folder containing the series")
parser.add_argument("--numimages", type=int, help="Number of images you want, defaults to all images")
parser.add_argument("--outfolder", type=str,default=None, help="Root of the folder that is going to contain the images and labels")
parser.add_argument("--radius", type=int,  default=6, help="Radious of the shpere being created in mm")
args = parser.parse_args()

TRAIN_LABELS_ROOT = args.labels
TRAIN_LOCALIZERS_ROOT = args.localizers
SERIES_ROOT = args.series
NUMBER_OF_IMAGES = args.numimages
OUT_ROOT = args.outfolder
SPHERE_RADIUS = args.radius



def _require_nibabel() -> Any:
    if nib is None:
        raise ImportError("nibabel is required for NIfTI output but is not installed")
    return nib


def _require_pydicom() -> Any:
    if pydicom is None:
        raise ImportError("pydicom is required for DICOM loading but is not installed")
    return pydicom


def _has_attr(ds: Any, name: str) -> bool:
    return hasattr(ds, name)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _sort_dicoms_fallback(dicoms: Sequence[Any]) -> List[Any]:
    dicoms = list(dicoms)
    if not dicoms:
        return []

    if all(_has_attr(ds, "ImagePositionPatient") for ds in dicoms):
        sorted_x = sorted(dicoms, key=lambda ds: float(ds.ImagePositionPatient[0]))
        sorted_y = sorted(dicoms, key=lambda ds: float(ds.ImagePositionPatient[1]))
        sorted_z = sorted(dicoms, key=lambda ds: float(ds.ImagePositionPatient[2]))

        diff_x = abs(float(sorted_x[-1].ImagePositionPatient[0]) - float(sorted_x[0].ImagePositionPatient[0]))
        diff_y = abs(float(sorted_y[-1].ImagePositionPatient[1]) - float(sorted_y[0].ImagePositionPatient[1]))
        diff_z = abs(float(sorted_z[-1].ImagePositionPatient[2]) - float(sorted_z[0].ImagePositionPatient[2]))

        if diff_x >= diff_y and diff_x >= diff_z:
            return sorted_x
        if diff_y >= diff_x and diff_y >= diff_z:
            return sorted_y
        return sorted_z

    return sorted(dicoms, key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))


def sort_dicoms_for_volume(dicoms: Sequence[Any]) -> List[Any]:
    dicoms = list(dicoms)
    if d2n_common is not None:
        try:
            if all(_has_attr(ds, "ImagePositionPatient") for ds in dicoms):
                return list(d2n_common.sort_dicoms(dicoms))
        except Exception:
            pass
    return _sort_dicoms_fallback(dicoms)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero-length vector")
    return vec / norm


def estimate_uniform_slice_spacing_mm(sorted_dicoms: Sequence[Any]) -> float:
    sorted_dicoms = list(sorted_dicoms)
    if not sorted_dicoms:
        return 1.0

    if len(sorted_dicoms) > 1 and all(_has_attr(ds, "ImagePositionPatient") for ds in sorted_dicoms):
        slice_positions = get_slice_positions_mm(sorted_dicoms)
        diffs = np.abs(np.diff(slice_positions))
        diffs = diffs[diffs > 1e-6]
        if diffs.size:
            return float(np.median(diffs))

    first = sorted_dicoms[0]
    if _has_attr(first, "SpacingBetweenSlices"):
        return abs(_safe_float(first.SpacingBetweenSlices, 1.0))
    if _has_attr(first, "SliceThickness"):
        return abs(_safe_float(first.SliceThickness, 1.0))
    return 1.0


def get_slice_positions_mm(sorted_dicoms: Sequence[Any]) -> np.ndarray:
    sorted_dicoms = list(sorted_dicoms)
    if not sorted_dicoms:
        return np.array([], dtype=float)

    if all(_has_attr(ds, "ImagePositionPatient") for ds in sorted_dicoms):
        positions = np.asarray([np.asarray(ds.ImagePositionPatient, dtype=float) for ds in sorted_dicoms], dtype=float)

        if all(_has_attr(ds, "ImageOrientationPatient") for ds in sorted_dicoms):
            iop = np.asarray(sorted_dicoms[0].ImageOrientationPatient, dtype=float)
            row_dir = _normalize(iop[:3])
            col_dir = _normalize(iop[3:6])
            normal = _normalize(np.cross(row_dir, col_dir))
        else:
            if len(sorted_dicoms) == 1:
                normal = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                delta = positions[-1] - positions[0]
                normal = np.array([0.0, 0.0, 1.0], dtype=float) if np.linalg.norm(delta) == 0 else _normalize(delta)

        return positions @ normal

    dz = estimate_uniform_slice_spacing_mm(sorted_dicoms)
    return np.arange(len(sorted_dicoms), dtype=float) * dz


def _apply_basic_dicom_scaling(data: np.ndarray, ds: Any) -> np.ndarray:
    slope = _safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
    intercept = _safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
    if slope == 1.0 and intercept == 0.0:
        return data
    return data.astype(np.float32, copy=False) * slope + intercept


def _get_volume_pixeldata_fallback(sorted_dicoms: Sequence[Any]) -> np.ndarray:
    slices = []
    combined_dtype = None

    for ds in sorted_dicoms:
        arr = np.asarray(ds.pixel_array)
        arr = _apply_basic_dicom_scaling(arr, ds)
        arr = arr[np.newaxis, :, :]
        slices.append(arr)
        combined_dtype = arr.dtype if combined_dtype is None else np.promote_types(combined_dtype, arr.dtype)

    if not slices:
        raise ValueError("No slices to stack")

    vol_zyx = np.concatenate([s.astype(combined_dtype, copy=False) for s in slices], axis=0)
    return np.transpose(vol_zyx, (2, 1, 0))


def get_volume_pixeldata_xyz(sorted_dicoms: Sequence[Any]) -> np.ndarray:
    if d2n_common is not None:
        try:
            return np.asarray(d2n_common.get_volume_pixeldata(list(sorted_dicoms)))
        except Exception:
            pass
    return _get_volume_pixeldata_fallback(sorted_dicoms)


def _orthogonalize_affine(affine: np.ndarray) -> np.ndarray:
    """Forces the rotation/scaling part of the affine to be orthonormal."""
    new_affine = affine.copy()
    # Extract the X and Y direction vectors
    c1 = new_affine[:3, 0]
    c2 = new_affine[:3, 1]

    # Get their norms (spacings)
    n1 = np.linalg.norm(c1)
    n2 = np.linalg.norm(c2)

    # Standardize X, then compute Y to be 90 degrees to X
    c1_unit = c1 / n1
    c2_unit = c2 / n2

    # Compute a perfectly perpendicular Z via cross product
    c3_unit = np.cross(c1_unit, c2_unit)
    c3_unit /= np.linalg.norm(c3_unit)

    # Re-project Y to be perpendicular to X (Gram-Schmidt)
    c2_unit_perp = np.cross(c3_unit, c1_unit)

    # Extract original Z-spacing from the old Z column
    n3 = np.linalg.norm(new_affine[:3, 2])

    # Reconstruct the rotation part
    new_affine[:3, 0] = c1_unit * n1
    new_affine[:3, 1] = c2_unit_perp * n2
    new_affine[:3, 2] = c3_unit * n3

    return new_affine


def _create_affine_fallback(sorted_dicoms: Sequence[Any]) -> Tuple[np.ndarray, float]:
    sorted_dicoms = list(sorted_dicoms)
    if not sorted_dicoms:
        raise ValueError("No dicoms available")
    if not all(_has_attr(ds, "ImageOrientationPatient") and _has_attr(ds, "ImagePositionPatient") for ds in sorted_dicoms):
        raise ValueError("Cannot build affine without ImageOrientationPatient and ImagePositionPatient")

    # Standard DICOM orientation vectors
    iop = np.asarray(sorted_dicoms[0].ImageOrientationPatient, dtype=float)
    orient_x = iop[:3]
    orient_y = iop[3:6]

    # Pixel spacings
    delta_r = float(sorted_dicoms[0].PixelSpacing[0])
    delta_c = float(sorted_dicoms[0].PixelSpacing[1])

    # Slice spacing logic
    image_pos = np.asarray(sorted_dicoms[0].ImagePositionPatient, dtype=float)
    if len(sorted_dicoms) == 1:
        step_norm = _safe_float(getattr(sorted_dicoms[0], "SliceThickness", 1.0), 1.0)
        # Use cross product for Z direction if only one slice
        orient_z = np.cross(orient_x, orient_y)
        step = orient_z * step_norm
    else:
        last_image_pos = np.asarray(sorted_dicoms[-1].ImagePositionPatient, dtype=float)
        step = (last_image_pos - image_pos) / (len(sorted_dicoms) - 1)
        step_norm = float(np.linalg.norm(step))

    if step_norm == 0.0:
         raise ValueError("NOT_A_VOLUME")

    # Build affine matrix (LPS to RAS conversion: negate first two components of vectors and origin)
    # We negate x and y to map DICOM's Left-Posterior-Superior to NIfTI's Right-Anterior-Superior
    affine = np.eye(4)
    affine[:3, 0] = -orient_x * delta_c
    affine[:3, 1] = -orient_y * delta_r
    affine[:3, 2] = -step
    affine[:3, 3] = -image_pos

    # Handle the flipping of the 3rd row for coordinate consistency
    affine[2, :3] *= -1
    affine[2, 3] *= -1

    # CRITICAL: Clean up the matrix so SimpleITK is happy
    affine = _orthogonalize_affine(affine)

    return affine, step_norm


def create_affine_from_sorted_dicoms(sorted_dicoms: Sequence[Any]) -> Tuple[np.ndarray, float]:
    if d2n_common is not None:
        try:
            return d2n_common.create_affine(list(sorted_dicoms))
        except Exception:
            pass
    return _create_affine_fallback(sorted_dicoms)


def build_reference_nifti_from_dicoms(sorted_dicoms: Sequence[Any]) -> Any:
    nib_mod = _require_nibabel()
    data_xyz = get_volume_pixeldata_xyz(sorted_dicoms)
    affine, _ = create_affine_from_sorted_dicoms(sorted_dicoms)
    # Final safety check on the affine
    affine = _orthogonalize_affine(affine)
    return nib_mod.Nifti1Image(data_xyz, affine)


def parse_coordinates(raw: Any) -> Dict[str, float]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = ast.literal_eval(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"coordinates must parse to a dict, got {type(parsed)!r}")
        return parsed
    raise TypeError(f"Unsupported coordinates type: {type(raw)!r}")


def paint_sphere_mm_variable_z(
    mask_zyx: np.ndarray,
    center_z: int,
    center_y: int,
    center_x: int,
    radius_mm: float,
    slice_positions_mm: Sequence[float],
    dy_mm: float,
    dx_mm: float,
    value: int = 1,
) -> None:
    z_count, y_count, x_count = mask_zyx.shape
    if not (0 <= center_z < z_count and 0 <= center_y < y_count and 0 <= center_x < x_count):
        return

    slice_positions_mm = np.asarray(slice_positions_mm, dtype=float)
    if slice_positions_mm.shape != (z_count,):
        raise ValueError(f"slice_positions_mm must have shape ({z_count},), got {slice_positions_mm.shape}")

    z_dist_mm = np.abs(slice_positions_mm - slice_positions_mm[center_z])
    candidate_z = np.where(z_dist_mm <= radius_mm + 1e-8)[0]
    if candidate_z.size == 0:
        return

    max_dy = int(np.ceil(radius_mm / dy_mm)) if dy_mm > 0 else 0
    max_dx = int(np.ceil(radius_mm / dx_mm)) if dx_mm > 0 else 0

    y_min = max(0, center_y - max_dy)
    y_max = min(y_count, center_y + max_dy + 1)
    x_min = max(0, center_x - max_dx)
    x_max = min(x_count, center_x + max_dx + 1)

    yy_mm = (np.arange(y_min, y_max, dtype=float) - float(center_y)) * dy_mm
    xx_mm = (np.arange(x_min, x_max, dtype=float) - float(center_x)) * dx_mm
    yy2 = yy_mm[:, None] ** 2
    xx2 = xx_mm[None, :] ** 2

    for z in candidate_z:
        r_xy2 = radius_mm ** 2 - z_dist_mm[z] ** 2
        if r_xy2 < 0:
            continue
        disk = (yy2 + xx2) <= (r_xy2 + 1e-8)
        sub = mask_zyx[z, y_min:y_max, x_min:x_max]
        sub[disk] = value


def _iter_candidate_dicom_files(series_dir: Path) -> List[Path]:
    series_dir = Path(series_dir)
    preferred = sorted(p for p in series_dir.glob("*.dcm") if p.is_file() and not p.name.startswith((".", "._")))
    if preferred:
        return preferred
    return sorted(p for p in series_dir.iterdir() if p.is_file() and not p.name.startswith((".", "._")))


def load_single_series_dicoms(series_dir: Path) -> List[Any]:
    pydicom_mod = _require_pydicom()
    ds_list = []
    for file_path in _iter_candidate_dicom_files(Path(series_dir)):
        try:
            ds_list.append(pydicom_mod.dcmread(str(file_path)))
        except Exception:
            continue
    return ds_list


def get_positive_cta_series_paths(train_csv: Any, series_root: Path) -> List[Path]:
    import pandas as pd
    if isinstance(train_csv, (str, Path)):
        train_df = pd.read_csv(train_csv)
    else:
        train_df = train_csv.copy()
    required = {"Modality", "Aneurysm Present", "SeriesInstanceUID"}
    missing = required.difference(train_df.columns)
    if missing:
        raise ValueError(f"train_csv is missing required columns: {sorted(missing)}")
    series_root = Path(series_root)
    positive = train_df.loc[train_df["Modality"].eq("CTA") & train_df["Aneurysm Present"].eq(1), "SeriesInstanceUID"].astype(str)
    return [series_root / uid for uid in positive.unique() if (series_root / uid).exists()]


def _rows_by_sop(train_localizers: Any) -> Mapping[str, List[Any]]:
    mapping: Dict[str, List[Any]] = {}
    for _, row in train_localizers.iterrows():
        sop = str(row["SOPInstanceUID"])
        mapping.setdefault(sop, []).append(row)
    return mapping


def _series_output_paths(series_name: str, ref_root: Path, label_root: Path) -> Tuple[Path, Path]:
    ref_path = Path(ref_root) / series_name / f"{series_name}_ref.nii.gz"
    label_path = Path(label_root) / series_name / f"{series_name}_label.nii.gz"
    return ref_path, label_path


def _remove_if_exists(path: Optional[Path]) -> None:
    if path is None: return
    try:
        if path.exists() or path.is_symlink():
            path.unlink()
    except FileNotFoundError:
        pass


def _cleanup_empty_parents(path: Path, stop_at: Path) -> None:
    current = path.parent
    stop_at = Path(stop_at).resolve()
    while True:
        try:
            current_resolved = current.resolve()
            if current_resolved == stop_at or stop_at not in current_resolved.parents:
                break
            current.rmdir()
            current = current.parent
        except (OSError, Exception):
            break


def _remove_series_outputs(series_name: str, ref_root: Path, label_root: Path) -> None:
    ref_path, label_path = _series_output_paths(series_name, ref_root, label_root)
    _remove_if_exists(ref_path)
    _remove_if_exists(label_path)
    _cleanup_empty_parents(ref_path, Path(ref_root))
    _cleanup_empty_parents(label_path, Path(label_root))


def _write_paired_nifti(ref_img: Any, mask_xyz: np.ndarray, ref_path: Path, label_path: Path) -> None:
    nib_mod = _require_nibabel()
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tmp_ref_path = tmpdir_path / ref_path.name
        tmp_label_path = tmpdir_path / label_path.name
        nib_mod.save(ref_img, str(tmp_ref_path))
        mask_header = ref_img.header.copy()
        mask_header.set_data_dtype(np.uint8)
        mask_img = nib_mod.Nifti1Image(mask_xyz.astype(np.uint8, copy=False), ref_img.affine, header=mask_header)
        nib_mod.save(mask_img, str(tmp_label_path))
        for p, tmp_p in [(ref_path, tmp_ref_path), (label_path, tmp_label_path)]:
            if p.exists():
                _remove_if_exists(p)
            shutil.move(str(tmp_p), str(p))


def _sync_paired_outputs(ref_root: Path, label_root: Path) -> None:
    ref_root, label_root = Path(ref_root), Path(label_root)
    def _collect(root: Path, suffix: str) -> Dict[str, Path]:
        if not root.exists(): return {}
        return {path.name[:-len(suffix)]: path for path in root.rglob("*.nii.gz") if path.is_file() and path.name.endswith(suffix)}
    ref_files = _collect(ref_root, "_ref.nii.gz")
    label_files = _collect(label_root, "_label.nii.gz")
    for series_name in set(ref_files) | set(label_files):
        if series_name not in ref_files or series_name not in label_files:
            _remove_series_outputs(series_name, ref_root, label_root)


def process_series_paths(paths: Iterable[Any], train_localizers: Any, ref_root: Optional[Path] = None, label_root: Optional[Path] = None, sphere_radius_mm: float = 1.5) -> None:
    _require_nibabel()
    ref_root = Path(ref_root or Path.home() / "Desktop" / "refs3")
    label_root = Path(label_root or Path.home() / "Desktop" / "labels3")
    rows_by_sop = _rows_by_sop(train_localizers)
    for path in paths:
        series_dir = Path(path)
        if not series_dir.exists() or not series_dir.is_dir(): continue
        series_name = series_dir.name
        ref_path, label_path = _series_output_paths(series_name, ref_root, label_root)
        ds_list = load_single_series_dicoms(series_dir)
        if not ds_list or (len(ds_list) == 1 and _has_attr(ds_list[0], "NumberOfFrames")):
            _remove_series_outputs(series_name, ref_root, label_root)
            continue
        try:
            sorted_ds = sort_dicoms_for_volume(ds_list)
            if not sorted_ds: raise ValueError()
            first = sorted_ds[0]
            dy_mm, dx_mm = float(first.PixelSpacing[0]), float(first.PixelSpacing[1])
            y_size, x_size = int(first.Rows), int(first.Columns)
            z_size = len(sorted_ds)
            mask_zyx = np.zeros((z_size, y_size, x_size), dtype=np.uint8)
            sop_to_z = {str(ds.SOPInstanceUID): i for i, ds in enumerate(sorted_ds) if _has_attr(ds, "SOPInstanceUID")}
            slice_positions_mm = get_slice_positions_mm(sorted_ds)
            for sop, z in sop_to_z.items():
                for row in rows_by_sop.get(sop, []):
                    xy = parse_coordinates(row["coordinates"])
                    x, y = int(round(float(xy["x"]))), int(round(float(xy["y"])))
                    if 0 <= x < x_size and 0 <= y < y_size:
                        paint_sphere_mm_variable_z(mask_zyx, z, y, x, sphere_radius_mm, slice_positions_mm, dy_mm, dx_mm)
            mask_xyz = np.transpose(mask_zyx, (2, 1, 0))
            ref_img = build_reference_nifti_from_dicoms(sorted_ds)
            if tuple(ref_img.shape) != tuple(mask_xyz.shape): raise ValueError()
            _write_paired_nifti(ref_img, mask_xyz, ref_path, label_path)
        except Exception:
            _remove_series_outputs(series_name, ref_root, label_root)
            continue
    _sync_paired_outputs(ref_root, label_root)


import pandas as pd
from pathlib import Path

train_labels = pd.read_csv(TRAIN_LABELS_ROOT)
train_localizers = pd.read_csv(TRAIN_LOCALIZERS_ROOT)

series_root = Path(SERIES_ROOT)
paths = get_positive_cta_series_paths(train_labels, series_root)
if NUMBER_OF_IMAGES:
    paths = paths[:NUMBER_OF_IMAGES]

print(f"Found {len(paths)} positive CTA series")

process_series_paths(
    paths,
    train_localizers,
    ref_root= Path(OUT_ROOT) / "refs",
    label_root= Path(OUT_ROOT)/ "labels",
    sphere_radius_mm=SPHERE_RADIUS,
)

