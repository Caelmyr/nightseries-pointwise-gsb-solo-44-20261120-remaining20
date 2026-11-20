"""Catalog, time coverage, and calibration diagnosis HTTP endpoints."""
from flask import jsonify, request

from .astrometry import (
    MAX_ANCHORS, MAX_CATALOG, catalog_rows, describe_field, fit_sky_solution,
    pixel_from_sky, reference_projection, sky_from_pixel,
)
from .contact import contact_sheet
from .detector import as_masks, detector_map, detector_results
from .domain import MAX_MASKS, DomainError, dimensions_of, identifier
from .exports import csv_bytes
from .guiding import guide_report
from .platecheck import audit_plates, draw_plate_differences
from .observing import LOG_COLUMNS, coverage_report, draft_log_patch
from .sheets import read_sheet

CATALOG_COLUMNS = ("label", "ra", "dec", "magnitude")
ANCHOR_COLUMNS = ("label", "x", "y", "ra", "dec", "weight")


def catalog_projection(document, input_rows):
    dims = dimensions_of(document)
    projection = reference_projection(document)
    if projection is None:
        raise DomainError("参考曝光没有天球坐标，请导入带 TAN 头的图像或先拟合定标星")
    if input_rows and projection.system != "ICRS":
        raise DomainError("本地星表使用 ICRS；当前 FITS 参考坐标系不可直接混用")
    return {
        "revision": document["revision"],
        "field": describe_field(projection, dims),
        "rows": [pixel_from_sky(projection, row, dims) for row in input_rows],
    }


def catalog_csv(result):
    cols = ["label", "ra_deg", "dec_deg", "mag", "ref_x_px", "ref_y_px", "coverage"]
    return csv_bytes(cols, (
        [row["label"], row["ra"], row["dec"], row["magnitude"], row["x"], row["y"], row["state"]]
        for row in result["rows"]
    ))


def log_csv(document):
    return csv_bytes(LOG_COLUMNS, (
        [row["filename"], row["started"], row["exposure"], row["band"], row["notes"], "true" if row["enabled"] else "false"]
        for row in document["frames"]
    ))


def coverage_csv(report):
    columns = ["kind", "frame", "filter", "started_utc", "ended_utc", "duration_seconds", "state", "included_frames"]
    records = []
    for row in report["frames"]:
        records.append([
            "exposure", row["filename"], row["band"], row["started"], row["ended"], row["exposure"],
            "enabled", row["frame"],
        ])
    for row in report["chunks"]:
        records.append([
            "coverage", None, report["band"], row["start"], row["end"], row["seconds"],
            "shutter_open", " ".join(row["frames"]),
        ])
    for row in report["gaps"]:
        records.append([
            "gap", None, report["band"], row["from"], row["to"], row["seconds"],
            "long" if row["large"] else "expected", row["before_frame"],
        ])
    for row in report["overlaps"]:
        records.append([
            "overlap", None, report["band"], None, None, row["seconds"],
            "check_timestamps", f"{row['with']} {row['frame']}",
        ])
    return csv_bytes(columns, records)


def guide_log(result):
    fields = ("filename", "timestamp", "band", "state", "native_dx", "native_dy", "residual_dx", "residual_dy",
              "x", "y", "snr", "fwhm_x", "fwhm_y", "centroid_uncertainty", "outside_limit")
    return csv_bytes(fields, ([row.get(key) for key in fields] for row in result["rows"]))


def register_field_tools(api, service):
    # All feature groups share the workbench's request and response contract.
    from .web import attachment, image_response, request_data, verify_computed, version_for

    store = service.store

    @api.get("/series/<sid>/sky")
    def sky_information(sid):
        doc = store.get(sid)
        projection = reference_projection(doc)
        if projection is None:
            return jsonify(revision=doc["revision"], field=None, plate=doc.get("plate"), catalog=doc.get("catalog", []))
        result = catalog_projection(doc, doc.get("catalog", []))
        result["plate"] = doc.get("plate")
        if request.args.get("format") == "csv":
            return attachment(catalog_csv(result), "text/csv; charset=utf-8", "sky-catalog.csv")
        return jsonify(result)

    @api.get("/series/<sid>/sky/pixel")
    def sky_coordinate(sid):
        doc = store.get(sid)
        verify_computed(doc, request.args)
        projection = reference_projection(doc)
        if projection is None:
            raise DomainError("参考影像没有可用天球坐标")
        return jsonify(sky_from_pixel(projection, request.args.get("x"), request.args.get("y"), dimensions_of(doc)))

    @api.post("/series/<sid>/sky/plate")
    def try_plate(sid):
        options = request_data()
        doc = store.get(sid)
        verify_computed(doc, options)
        if not doc["reference"]:
            raise DomainError("请先指定参考影像")
        entries = options.get("anchors")
        if "csv" in options:
            entries = [
                {**row, "weight": row.get("weight") or 1}
                for _, row in read_sheet(
                    options["csv"], ANCHOR_COLUMNS, {"label", "x", "y", "ra", "dec"}, MAX_ANCHORS
                )
            ]
        result = fit_sky_solution(entries, options, dimensions_of(doc))
        if options.get("apply") is True:
            doc = store.set_plate(sid, version_for(options), result)
        return jsonify(result=result, document=doc)

    @api.delete("/series/<sid>/sky/plate")
    def reset_plate(sid):
        return jsonify(store.set_plate(sid, version_for(request.args), None))

    @api.put("/series/<sid>/sky/catalog")
    def import_catalog(sid):
        body = request_data()
        if "csv" in body:
            entries = [row for _, row in read_sheet(body["csv"], CATALOG_COLUMNS, {"label", "ra", "dec"}, MAX_CATALOG)]
        else:
            entries = body.get("rows")
        stored = store.set_catalog(sid, version_for(body), catalog_rows(entries))
        return jsonify(stored)

    @api.post("/series/<sid>/sky/audit")
    def sky_audit(sid):
        options = request_data()
        document = store.get(sid)
        verify_computed(document, options)
        report = audit_plates(document, options)
        if options.get("format") == "image":
            selected = options.get("frame") or report["worst"]
            image = draw_plate_differences(report, identifier(selected, "天球复核曝光"))
            return image_response(image)
        if options.get("format") == "csv":
            fields = ("filename", "band", "state", "used", "out_of_frame", "median_error", "maximum_error", "rms_error")
            data = csv_bytes(fields, ([row.get(key) for key in fields] for row in report["rows"]))
            return attachment(data, "text/csv; charset=utf-8", "reference-plate-audit.csv")
        return jsonify(report)

    @api.get("/series/<sid>/timeline")
    def read_timeline(sid):
        doc = store.get(sid)
        verify_computed(doc, request.args)
        report = coverage_report(doc, request.args)
        if request.args.get("format") == "csv":
            return attachment(coverage_csv(report), "text/csv; charset=utf-8", "observing-coverage.csv")
        return jsonify(report)

    @api.get("/series/<sid>/timeline/log")
    def export_observing_log(sid):
        return attachment(log_csv(store.get(sid)), "text/csv; charset=utf-8", "observation-log.csv")

    @api.post("/series/<sid>/timeline/log")
    def inspect_observing_log(sid):
        options = request_data()
        doc = store.get(sid)
        verify_computed(doc, options)
        draft = draft_log_patch(doc, options.get("csv"))
        if options.get("apply") is True:
            if draft["issues"]:
                raise DomainError("请修正存在异议的日志行后再应用")
            doc = store.apply_log(sid, version_for(options), draft["updates"])
        return jsonify(result=draft, document=doc)

    @api.post("/series/<sid>/detector")
    def monitor_detector(sid):
        options = request_data()

        def compute():
            images = service.images(sid)
            verify_computed(images.document, options)
            report, center, flag = detector_results(images, options)
            if options.get("format") == "image":
                return image_response(detector_map(center, flag, report))
            payload = {"report": report}
            if options.get("apply") is True:
                selected = as_masks(report, options.get("selected"), maximum=MAX_MASKS)
                payload["document"] = store.append_detector_masks(sid, version_for(options), selected)
            return jsonify(payload)
        return service.compute(compute)

    @api.post("/series/<sid>/guide")
    def verify_guide(sid):
        options = request_data()

        def compute():
            images = service.images(sid)
            verify_computed(images.document, options)
            result = guide_report(images, options)
            if options.get("format") == "csv":
                return attachment(guide_log(result), "text/csv; charset=utf-8", "guiding-review.csv")
            return jsonify(result)
        return service.compute(compute)

    @api.post("/series/<sid>/contact")
    def render_contact_sheet(sid):
        options = request_data()

        def compute():
            images = service.images(sid)
            verify_computed(images.document, options)
            image, report = contact_sheet(images, options)
            if options.get("format") == "json":
                return jsonify(report)
            if options.get("format") == "download":
                return attachment(image, "image/png", "exposure-contact-sheet.png")
            return image_response(image, X_Contact_Frame_Count=report["count"], X_Contact_Visible_Count=report["displayed"])
        return service.compute(compute)
