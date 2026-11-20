"""HTTP entry points for image inspection adjuncts."""
from flask import jsonify

from .background import render_sky, sky_surface
from .cutout import cutout_zip
from .domain import enum
from .exports import csv_bytes
from .footprint import draw_coverage, footprint_plan
from .gain import gain_report
from .growth import flux_growth
from .noise_curve import TILE_REJECTIONS, noise_curve_report
from .pairing import pair_report
from .registration import preview_stream
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


def noise_level_csv(report):
    columns = [
        "level_index", "median_net_adu", "level_spread_adu", "state", "state_reason",
        "frame_count", "independent_frames_used", "pairs_total", "pairs_measured",
        "included_flat_filenames", "max_saturated_fraction",
        "gain_median_e_per_adu", "gain_sigma", "noise_median_adu", "noise_sigma_adu",
        "expected_noise_adu", "gain_deviation", "noise_deviation",
    ]

    def rows():
        for level in report["levels"]:
            yield [
                level["index"], level["median_net_adu"], level["level_spread_adu"], level["state"],
                level["state_reason"], level["frame_count"], level["independent_frames"],
                level["pair_count_total"], level["pair_count_measured"],
                " | ".join(frame["filename"] for frame in level["frames"]),
                level["max_saturated_fraction"], level["gain_median"], level["gain_sigma"],
                level["noise_adu"], level["noise_sigma"], level["expected_noise_adu"],
                level["gain_deviation"], level["noise_deviation"],
            ]

    return csv_bytes(columns, rows())


def noise_pair_csv(report):
    columns = [
        "level_index", "state", "first_flat", "second_flat", "exposure_seconds",
        "net_level_adu", "tiles_kept", "tiles_total", "gain_median_e_per_adu",
        "gain_sigma", "noise_median_adu", "variance_median_adu2", "rejection_reasons",
    ]

    def rows():
        for level in report["levels"]:
            for pair in level["pairs"]:
                reasons = " | ".join(
                    f"{TILE_REJECTIONS.get(code, code)}:{count}"
                    for code, count in sorted(pair["rejection_reasons"].items())
                )
                yield [
                    level["index"], pair["state"], pair["first"], pair["second"], pair["exposure"],
                    pair["net_level_adu"], pair["tiles_kept"], pair["tiles_total"],
                    pair["gain_median"], pair["gain_sigma"], pair["noise_adu"],
                    pair["variance_median"], reasons,
                ]

    return csv_bytes(columns, rows())


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

    @api.post("/series/<sid>/noise-curve")
    def flat_noise_curve(sid):
        def compute(images, options):
            result = noise_curve_report(images, options)
            if options.get("format") == "level_csv":
                return attachment(noise_level_csv(result), "text/csv; charset=utf-8", "flat-noise-levels.csv")
            if options.get("format") == "pair_csv":
                return attachment(noise_pair_csv(result), "text/csv; charset=utf-8", "flat-noise-pairs.csv")
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
