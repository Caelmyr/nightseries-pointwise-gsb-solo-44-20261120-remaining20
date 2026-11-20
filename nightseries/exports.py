"""Portable workbooks and CSV files derived from the same observing results."""
import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from .astrometry import catalog_rows, fit_sky_solution
from .domain import (
    MAX_APERTURES, MAX_ASSET_BYTES, MAX_FRAMES, MAX_MASKS, MAX_SERIES_BYTES, MAX_SNAPSHOTS,
    DomainError, alignment_value, aperture_value, boolean, calibration_settings,
    identifier, integer, iso_time, list_value, mask_value, new_document, number,
    object_value, photometry_settings, stretch_value, text,
)
from .store import Store

SCHEMA_VERSION = 1
ARCHIVE_LIMIT = 120 * 1024 * 1024
MANIFEST_LIMIT = 8 * 1024 * 1024


def spreadsheet_cell(value):
    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@"}:
        return "'" + value
    return value


def csv_bytes(columns, rows):
    text_buffer = StringIO(newline="")
    writer = csv.writer(text_buffer)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(spreadsheet_cell(value) for value in row)
    return text_buffer.getvalue().encode("utf-8-sig")


def photometry_csv(curve):
    columns = [
        "filename", "midpoint_utc", "exposure_seconds", "filter", "source", "role",
        "state", "native_x", "native_y", "coverage", "flux_adu", "flux_per_second",
        "uncertainty_per_second", "sky_adu", "sky_noise_adu", "differential_mag",
        "differential_error", "differential_state",
    ]
    rows = []
    for frame in curve["rows"]:
        for measurement in frame["measurements"]:
            role = measurement["role"]
            differential = (frame.get(role) or {}) if role != "comparison" else {}
            sky = measurement.get("background") or {}
            rows.append([
                frame["filename"], frame["midpoint"], frame["exposure"], frame["band"],
                measurement["label"], role, measurement["state"], measurement["native_x"],
                measurement["native_y"], measurement["coverage"], measurement["flux"],
                measurement["flux_per_second"], measurement["uncertainty_per_second"],
                sky.get("level"), sky.get("noise"), differential.get("magnitude"),
                differential.get("error"), differential.get("state"),
            ])
    return csv_bytes(columns, rows)


def quality_csv(report):
    columns = [
        "filename", "started_utc", "filter", "star_count", "fwhm_px", "elongation_ratio",
        "sky_median_adu", "sky_noise_adu", "sky_spread_adu", "masked_fraction",
        "saturated_fraction", "dx_px", "dy_px", "flags",
    ]
    rows = (
        [
            row["filename"], row["started"], row["band"], row["stars_used"],
            row["median_fwhm"], row["median_elongation"],
            (row["sky"] or {}).get("median"), (row["sky"] or {}).get("sigma"),
            row["sky_spread"], row["masked_fraction"], row["saturated_fraction"],
            *(row["offset"] or (None, None)), " ".join(row["flags"]),
        ]
        for row in report["rows"]
    )
    return csv_bytes(columns, rows)


def profile_csv(profile):
    return csv_bytes(
        ["distance_px", "x", "y", "value_adu", "saturated"],
        ([row["distance"], row["x"], row["y"], row["value"], row["saturated"]] for row in profile["rows"]),
    )


def tracklet_csv(result):
    columns = [
        "filename", "started_utc", "midpoint_utc", "exposure_seconds", "filter",
        "state", "reason", "seed_frame", "extrapolated",
        "pred_ref_x", "pred_ref_y", "measured_ref_x", "measured_ref_y",
        "native_x", "native_y", "snr", "flux_adu", "fwhm_px", "offset_distance_px",
        "candidates", "note",
    ]
    rows = []
    for row in result["rows"]:
        measured = row["measured"] or {}
        rows.append([
            row["filename"], row["started"], row["midpoint"], row["exposure"], row["band"],
            row["state"], row["reason"], "true" if row["seed"] else "false",
            "true" if row["extrapolated"] else "false",
            row["pred_x"], row["pred_y"],
            measured.get("ref_x"), measured.get("ref_y"),
            measured.get("native_x"), measured.get("native_y"),
            measured.get("snr"), measured.get("flux"), measured.get("fwhm"),
            measured.get("offset_distance"), len(row["candidates"]), row["note"],
        ])
    return csv_bytes(columns, rows)


def asset_name(frame):
    return f"exposures/{frame['id']}{Path(frame['filename']).suffix.lower()}"


def write_bundle(store, series_id):
    doc, images, snapshots = store.bundle(series_id)
    target = BytesIO()
    with ZipFile(target, "w", ZIP_DEFLATED, compresslevel=3) as archive:
        manifest = {"schema": SCHEMA_VERSION, "document": doc, "snapshots": snapshots}
        manifest_body = json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode()
        if len(manifest_body) > MANIFEST_LIMIT:
            raise DomainError("归档说明文件超过交换容量，请移除部分保存结果")
        archive.writestr("observation.json", manifest_body)
        for entry in doc["frames"] + doc["calibration_frames"]:
            if entry["id"] not in images:
                raise DomainError("原曝光丢失，无法生成交换包", 409)
            archive.writestr(asset_name(entry), images[entry["id"]])
    return target.getvalue()


def read_member(archive, name, limit):
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise DomainError(f"观测包缺少成员 {name}") from exc
    if info.is_dir() or not 0 <= info.file_size <= limit:
        raise DomainError("观测包成员超过容量或不是文件")
    return archive.read(info)


def prepared_members(archive, doc):
    science = list_value(doc.get("frames"), "科学帧", MAX_FRAMES)
    calibration = list_value(doc.get("calibration_frames"), "校准帧", MAX_FRAMES)
    originals = science + calibration
    if len(originals) > MAX_FRAMES:
        raise DomainError("观测包的总帧数超过容量")
    result = []
    identifiers = set()
    size = 0
    for original in originals:
        row = object_value(original, "影像记录")
        frame_id = identifier(row.get("id"), "曝光")
        filename = text(row.get("filename"), "文件名", 140)
        if filename != Path(filename).name:
            raise DomainError("观测包的影像文件名必须在本级")
        if frame_id in identifiers:
            raise DomainError("观测包有重复曝光编号")
        identifiers.add(frame_id)
        member = asset_name({"id": frame_id, "filename": filename})
        body = read_member(archive, member, MAX_ASSET_BYTES)
        size += len(body)
        if size > MAX_SERIES_BYTES:
            raise DomainError("观测包的原曝光超过观测组容量")
        prepared = Store.prepare_asset(filename, body, row.get("kind"), row)
        actual = prepared[0]
        if actual["sha256"] != row.get("sha256") or actual["byte_size"] != row.get("byte_size"):
            raise DomainError("观测包的原曝光与清单校验不一致")
        actual["id"] = frame_id
        result.append(prepared)
    expected = {"observation.json"} | {asset_name(row) for row, _, _ in result}
    if expected != set(archive.namelist()):
        raise DomainError("观测包含未在清单中登记的成员")
    return result


def validated_saved_state(doc, prepared):
    dims = prepared[0][1] if prepared else None
    frames = {row["id"] for row, _, _ in prepared if row["kind"] == "science"}
    reference = doc.get("reference")
    if frames and identifier(reference, "基准帧") not in frames:
        raise DomainError("包中没有所登记的基准曝光")
    if reference and not frames:
        raise DomainError("无科学曝光的观测包不能登记基准")
    apertures = list_value(doc.get("apertures"), "孔径", MAX_APERTURES)
    masks = list_value(doc.get("masks"), "掩膜", MAX_MASKS)
    offsets = object_value(doc.get("alignments"), "对准偏移")
    if (apertures and not reference) or (masks and dims is None):
        raise DomainError("观测包中的区域缺少对应影像")
    safe_aps = []
    for row in apertures:
        item = aperture_value(row, dims)
        item["id"] = identifier(row.get("id"), "孔径")
        safe_aps.append(item)
    if len({row["id"] for row in safe_aps}) != len(safe_aps) or len({row["label"] for row in safe_aps}) != len(safe_aps):
        raise DomainError("观测包的孔径编号或标签重复")
    safe_masks = []
    for row in masks:
        item = mask_value(row, dims)
        item["id"] = identifier(row.get("id"), "掩膜")
        if item["frame"] is not None and item["frame"] not in frames:
            raise DomainError("观测包中的掩膜引用不存在的曝光")
        safe_masks.append(item)
    if len({row["id"] for row in safe_masks}) != len(safe_masks):
        raise DomainError("观测包的掩膜编号重复")
    safe_offsets = {}
    for frame, row in offsets.items():
        if frame not in frames or frame == reference:
            raise DomainError("观测包有无效的对准偏移")
        offset = alignment_value(row)
        if row.get("source") == "automatic":
            offset["source"] = "automatic"
            offset["score"] = number(row.get("score"), "对准相关值", 0, 1.001)
            ratio = row.get("peak_ratio")
            offset["peak_ratio"] = number(ratio, "候选峰比", 0, 1e12) if ratio is not None else None
        safe_offsets[frame] = offset
    plate = None
    if doc.get("plate") is not None:
        saved = object_value(doc.get("plate"), "定标图幅")
        if not reference or saved.get("reference") != reference:
            raise DomainError("定标图幅引用了不同的参考曝光")
        plate = fit_sky_solution(saved.get("anchors"), saved, dims)
        plate["reference"] = reference
    catalog = catalog_rows(doc.get("catalog", []))
    return {
        "reference": reference,
        "plate": plate,
        "catalog": catalog,
        "apertures": safe_aps,
        "masks": safe_masks,
        "alignments": safe_offsets,
        "notes": text(doc.get("notes", ""), "观测备注", 350, required=False),
        "photometry": photometry_settings(doc.get("photometry")),
        "calibration": calibration_settings(doc.get("calibration")),
        "stretch": stretch_value(doc.get("stretch")),
    }


def read_bundle(store, body):
    if not 0 < len(body) <= ARCHIVE_LIMIT:
        raise DomainError("观测包为空或超过 120 MiB")
    try:
        with ZipFile(BytesIO(body)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or not 1 <= len(names) <= MAX_FRAMES + 1:
                raise DomainError("观测包目录有重复成员或数量超出范围")
            manifest = object_value(json.loads(read_member(archive, "observation.json", MANIFEST_LIMIT)), "交换清单")
            if manifest.get("schema") != SCHEMA_VERSION:
                raise DomainError("不支持此观测包版本")
            incoming = object_value(manifest.get("document"), "观测组")
            document = new_document(
                text(incoming.get("name"), "观测组名称", 80),
                boolean(incoming.get("synthetic"), "合成样本"),
            )
            prepared = prepared_members(archive, incoming)
            state = validated_saved_state(incoming, prepared)
            snapshots = list_value(manifest.get("snapshots", []), "结果归档", MAX_SNAPSHOTS)
            entries = []
            for value in snapshots:
                row = object_value(value, "归档条目")
                entries.append({
                    "created": iso_time(row.get("created"), "归档时间"),
                    "label": row.get("label"),
                    "kind": row.get("kind"),
                    "revision": integer(row.get("revision"), "归档版本", 1, 2**52),
                    "result": object_value(row.get("result"), "归档结果"),
                })
    except (BadZipFile, UnicodeError, json.JSONDecodeError, RecursionError, RuntimeError, NotImplementedError) as exc:
        raise DomainError("不能读取该观测包或清单") from exc
    return store.create_populated(document, prepared, initial_state=state, snapshots=entries)
