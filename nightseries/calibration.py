"""Detector-space calibration with explicit pixel validity and provenance."""
from dataclasses import dataclass
from collections import defaultdict
import math
import warnings

import numpy as np

from .domain import DomainError, MAX_ACTIVE_PIXELS, dimensions_of
from .raster import read_image

CALIBRATION_NAMES = {"bias": "偏置", "dark": "暗场", "flat": "平场"}


@dataclass(frozen=True)
class Plane:
    values: np.ndarray
    valid: np.ndarray
    saturated: np.ndarray
    notes: tuple = ()

    @classmethod
    def raw(cls, image, saturation):
        samples = image.samples
        valid = np.isfinite(samples)
        saturated = valid & (samples >= saturation)
        return cls(samples, valid, saturated)


def finite_median(stack, axis=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        return np.nanmedian(stack, axis=axis)


def robust_summary(samples):
    valid = np.asarray(samples, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return {"median": None, "sigma": None, "count": 0}
    median = float(np.median(valid))
    sigma = 1.482602218505602 * float(np.median(np.abs(valid - median)))
    return {"median": median, "sigma": sigma, "count": int(valid.size)}


def masked_values(plane):
    return np.where(plane.valid & ~plane.saturated, plane.values, np.nan)


def master_median(values):
    if not values:
        return None
    if len(values) == 1:
        return values[0].copy()
    return finite_median(np.stack(values), axis=0)


def detector_mask(document, frame_id, shape):
    invalid = np.zeros(shape, dtype=bool)
    for region in document["masks"]:
        if region["frame"] and region["frame"] != frame_id:
            continue
        left = math.ceil(region["left"])
        top = math.ceil(region["top"])
        right = math.ceil(region["right"])
        bottom = math.ceil(region["bottom"])
        invalid[top:bottom, left:right] = True
    return invalid


def calibration_status(document):
    return {
        kind: [
            {"id": frame["id"], "filename": frame["filename"], "exposure": frame["exposure"], "band": frame["band"]}
            for frame in document["calibration_frames"]
            if frame["enabled"] and frame["kind"] == kind
        ] if switched else []
        for kind, switched in document["calibration"].items() if kind in CALIBRATION_NAMES
    }


class ObservationImages:
    """One coherent read from the store with request-local image and master reuse."""

    def __init__(self, document, bodies):
        self.document = document
        self.bodies = bodies
        self.dims = dimensions_of(document)
        self.rows = {row["id"]: row for row in document["frames"] + document["calibration_frames"]}
        self.inputs = {
            key: [self.rows[entry["id"]] for entry in values]
            for key, values in calibration_status(document).items()
        }
        self.settings = document["calibration"]
        self.saturation = document["photometry"]["saturation"]
        self._bias = None
        self._bias_ready = False
        self._dark = {}
        self._flat = {}
        self._calibration_pixels = sum(len(frames) for frames in self.inputs.values()) * self.dims.pixels
        if self._calibration_pixels > MAX_ACTIVE_PIXELS:
            raise DomainError("当前启用校准图超过单次计算像素预算，请挂起一部分校准图")

    def image(self, frame_id):
        if frame_id not in self.rows or frame_id not in self.bodies:
            raise DomainError("找不到这张曝光", 404)
        return read_image(self.rows[frame_id]["filename"], self.bodies[frame_id])

    def raw(self, frame_id, use_detector_mask=False):
        plane = Plane.raw(self.image(frame_id), self.saturation)
        if use_detector_mask:
            valid = plane.valid & ~detector_mask(self.document, frame_id, plane.values.shape)
            return Plane(plane.values, valid, plane.saturated)
        return plane

    def master_bias(self):
        if not self._bias_ready:
            self._bias = master_median([masked_values(self.raw(row["id"], True)) for row in self.inputs["bias"]])
            self._bias_ready = True
        return self._bias

    def dark_image(self, exposure):
        bias = self.master_bias()
        if bias is None:
            key = ("same", exposure)
        else:
            key = ("rate",)
        if key not in self._dark:
            darks = []
            for row in self.inputs["dark"]:
                if bias is None and not math.isclose(row["exposure"], exposure, rel_tol=1e-6, abs_tol=1e-6):
                    continue
                values = masked_values(self.raw(row["id"], True))
                if bias is not None:
                    values = (values - bias) / row["exposure"]
                darks.append(values)
            self._dark[key] = master_median(darks)
        master = self._dark[key]
        if master is None:
            return None, None
        if bias is None:
            return master, "同曝光暗场（含偏置）"
        return master * exposure, "暗电流率 × 曝光秒数"

    def linear_values(self, plane, exposure):
        values = np.where(plane.valid, plane.values, np.nan)
        bias = self.master_bias()
        dark, dark_method = self.dark_image(exposure)
        notes = []
        if bias is not None:
            values -= bias
            notes.append("偏置中位图")
        if dark is not None:
            values -= dark
            notes.append(dark_method)
        elif self.settings["dark"] and self.inputs["dark"]:
            notes.append("无匹配曝光的暗场")
        return values, notes

    def master_flat(self, band):
        if band in self._flat:
            return self._flat[band]
        flats = []
        for frame in self.inputs["flat"]:
            if frame["band"] != band:
                continue
            raw = self.raw(frame["id"], True)
            values, _ = self.linear_values(
                Plane(raw.values, raw.valid & ~raw.saturated, raw.saturated), frame["exposure"]
            )
            reference = robust_summary(values[values > 0])["median"]
            if reference is not None and reference > 0:
                flats.append(values / reference)
        combined = master_median(flats)
        if combined is not None:
            combined[combined < self.settings["flat_floor"]] = np.nan
        self._flat[band] = combined
        return combined

    def calibrated(self, frame_id):
        raw = self.raw(frame_id, use_detector_mask=True)
        frame = self.rows[frame_id]
        if frame["kind"] != "science":
            raise DomainError("校准预览只用于科学曝光")
        values, notes = self.linear_values(raw, frame["exposure"])
        flat = self.master_flat(frame["band"])
        if flat is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                values = values / flat
            notes.append("同滤镜归一平场")
        elif self.settings["flat"] and self.inputs["flat"]:
            notes.append("无有效同滤镜平场")
        valid = raw.valid & np.isfinite(values)
        values[~valid] = np.nan
        return Plane(values, valid, raw.saturated, tuple(notes))

    def master_panel(self):
        descriptions = {}
        bias = self.master_bias()
        if bias is not None:
            descriptions["bias"] = self.describe_master(bias, "ADU")
        bands = sorted({frame["band"] for frame in self.document["frames"]})
        for band in bands:
            flat = self.master_flat(band)
            if flat is not None:
                descriptions[f"flat:{band}"] = self.describe_master(flat, "归一响应")
        by_exposure = defaultdict(list)
        for frame in self.document["frames"]:
            by_exposure[frame["exposure"]].append(frame["filename"])
        descriptions["dark_applications"] = [
            {"exposure": exposure, "files": files, "method": self.dark_image(exposure)[1]}
            for exposure, files in sorted(by_exposure.items())
        ]
        return descriptions

    @staticmethod
    def describe_master(values, unit):
        result = robust_summary(values)
        result["unit"] = unit
        result["invalid"] = int(values.size - result["count"])
        return result
