"""HTTP entry points for image inspection adjuncts."""
from flask import jsonify

from .background import render_sky, sky_surface
from .cutout import cutout_zip
from .domain import enum
from .exports import csv_bytes
from .footprint import draw_coverage, footprint_plan
from .gain import gain_report
from .growth import flux_growth
from .pairing import pair_report
from .registration import preview_stream
from .shell import draw_shell, shell_csv, shell_profile
from .surface import draw_regions_on_reference, surface_brightness


def growth_csv(result):
    return csv_bytes(
        ["radius_px", "support", "fraction", "spread", "snr", "median_coverage"],
        ([level["radius"], level["count"], level["fraction"], level["fraction_spread"], level["median_snr"], level["median_overlap"]]
         for level in result["radii"]),
    )


def gain_csv(report):
    fields = ("first", "second", "x", "y", "center_adu", "variance_adu2", "gain", "pair_difference", "samples")
    return csv_bytes(fields, ([point[key] for key in fields] for point in report["estimates"]))


def surface_csv(report):
    names = ("filename", "timestamp", "band", "state", "surface_rate", "net_rate", "uncertainty", "target_fraction",
             "sky_fraction", "target_saturated", "sky_saturated", "target_pixels", "sky_pixels")
    return csv_bytes(names, ([row.get(key) for key in names] for row in report["rows"]))


def register_inspection_tools(api, service):
    from .web import attachment, image_response, request_data, verify_computed

    def run(sid, handler):
        options = request_data()

        def compute():
            images = service.images(sid)
            verify_computed(images.document, options)
            return handler(images, options)
        return service.compute(compute)

    @api.post("/series/<sid>/growth")
    def inspect_radii(sid):
        def compute(images, options):
            result = flux_growth(images, options)
            if options.get("format") == "csv":
                return attachment(growth_csv(result), "text/csv; charset=utf-8", "aperture-growth.csv")
            return jsonify(result)
        return run(sid, compute)

    @api.post("/series/<sid>/pair-inspection")
    def inspect_pair(sid):
        def compute(images, options):
            result, artifact = pair_report(images, options.get("a"), options.get("b"), options)
            if options.get("format") == "image":
                return image_response(artifact, X_Intensity_Slope=result["slope"], X_Background_Offset=result["offset"])
            return jsonify(result)
        return run(sid, compute)

    @api.post("/series/<sid>/gain")
    def repeated_flat_gain(sid):
        def compute(images, options):
            result = gain_report(images, options)
            if options.get("format") == "csv":
                return attachment(gain_csv(result), "text/csv; charset=utf-8", "detector-gain-check.csv")
            return jsonify(result)
        return run(sid, compute)

    @api.post("/series/<sid>/exposures/<frame>/background")
    def inspect_background(sid, frame):
        def compute(images, options):
            layer = enum(options.get("stream", "calibrated"), "背景图层", {"raw", "calibrated", "aligned"})
            report, model, residual = sky_surface(preview_stream(images, frame, layer), options)
            format = enum(options.get("format", "json"), "背景图输出", {"json", "sky", "residual"})
            if format != "json":
                return image_response(render_sky(model if format == "sky" else residual, format == "residual"))
            return jsonify(report)
        return run(sid, compute)

    @api.post("/series/<sid>/footprint")
    def common_field(sid):
        def compute(images, options):
            report, count, spec = footprint_plan(images, options)
            if options.get("format") == "image":
                return image_response(draw_coverage(count, report["used"], report["best"], spec))
            return jsonify(report)
        return run(sid, compute)

    @api.post("/series/<sid>/cutout")
    def export_native_stamps(sid):
        def compute(images, options):
            payload, report = cutout_zip(images, options)
            if options.get("format") == "json":
                return jsonify(report)
            return attachment(payload, "application/zip", "native-detector-cutouts.zip")
        return run(sid, compute)

    @api.post("/series/<sid>/surface")
    def extended_source(sid):
        def compute(images, options):
            format = enum(options.get("format", "json"), "扩展目标输出", {"json", "csv", "image"})
            if format == "image":
                return image_response(draw_regions_on_reference(images, options))
            result = surface_brightness(images, options)
            if format == "csv":
                return attachment(surface_csv(result), "text/csv; charset=utf-8", "extended-source-photometry.csv")
            return jsonify(result)
        return run(sid, compute)

    @api.post("/series/<sid>/shell")
    def shell_widths(sid):
        def compute(images, options):
            result, plane = shell_profile(images, options)
            mode = enum(options.get("format", "json"), "环壳输出", {"json", "csv", "image"})
            if mode == "csv":
                return attachment(shell_csv(result), "text/csv; charset=utf-8", "shell-widths.csv")
            if mode == "image":
                first, second = result["directions"]
                return image_response(
                    draw_shell(plane, result, images.document["stretch"]),
                    X_Shell_A_State=first["state"],
                    X_Shell_B_State=second["state"],
                    X_Shell_A_Width="" if first["width"] is None else first["width"],
                    X_Shell_B_Width="" if second["width"] is None else second["width"],
                )
            return jsonify(result)
        return run(sid, compute)
