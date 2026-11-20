"""Browser and file endpoints for a local NightSeries installation."""
from io import BytesIO
import mimetypes
from pathlib import Path
from threading import BoundedSemaphore

from flask import Blueprint, Flask, Response, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

from .calibration import ObservationImages, calibration_status
from .demo import SCENARIOS, make_demo
from .domain import (
    ASSET_KINDS, MAX_ASSET_BYTES, MAX_FRAMES, MAX_SERIES_BYTES, STREAMS, DomainError,
    enum, integer, list_value, object_value, text,
)
from .exports import (
    ARCHIVE_LIMIT, photometry_csv, profile_csv, quality_csv, read_bundle, tracklet_csv,
    write_bundle,
)
from .photometry import light_curve
from .preview import (
    difference_preview, inspect_pixel, raster_histogram, render_plane, sample_line,
)
from .quality import frame_quality, quality_report
from .raster import image_summary, make_fits
from .registration import preview_stream, run_registration
from .stacking import compute_stack, coverage_preview
from .store import Store
from .tracklet import tracklet_report


class Service:
    """Shared application services for independently registered endpoint groups."""

    def __init__(self, data_path):
        self.store = Store(Path(data_path) / "observations.sqlite3")
        self.compute_slots = BoundedSemaphore(2)

    def images(self, series_id):
        return ObservationImages(*self.store.assets(series_id))

    def compute(self, operation):
        if not self.compute_slots.acquire(blocking=False):
            raise DomainError("两个图像任务正在执行，请稍候重试", 503)
        try:
            return operation()
        finally:
            self.compute_slots.release()


def request_data():
    return object_value(request.get_json(), "请求内容")


def version_for(body):
    return integer(body.get("version"), "观测版本", 1, 2**52)


def query_version():
    return version_for(request.args)


def verify_computed(document, options):
    if "version" in options and document["revision"] != version_for(options):
        raise DomainError("观测组已变化，请刷新后再计算", 409)


def attachment(buffer, media, filename):
    return send_file(
        BytesIO(buffer), mimetype=media, as_attachment=True,
        download_name=filename, max_age=0, conditional=False,
    )


def image_response(data, **headers):
    response = Response(data, content_type="image/png")
    for name, value in headers.items():
        response.headers[name.replace("_", "-")] = str(value)
    return response


def javascript_option(value):
    return value == "1"


def view_plane(service, series_id, frame_id, params):
    images = service.images(series_id)
    verify_computed(images.document, params)
    stream = enum(params.get("stream", "raw"), "影像来源", STREAMS)
    return images, preview_stream(images, frame_id, stream)


def create_app(data_path):
    app = Flask(__name__, static_folder=None)
    app.config.update(
        MAX_CONTENT_LENGTH=ARCHIVE_LIMIT + 1024 * 1024,
        MAX_FORM_MEMORY_SIZE=1024 * 1024,
        MAX_FORM_PARTS=MAX_FRAMES + 20,
        JSON_SORT_KEYS=False,
    )
    service = Service(data_path)
    api = Blueprint("nightseries", __name__, url_prefix="/api")
    app.register_error_handler(DomainError, lambda exc: (jsonify(error=str(exc)), exc.status))
    app.register_error_handler(HTTPException, handle_http_error)

    @app.after_request
    def headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; frame-ancestors 'none'; base-uri 'none'",
        )
        return response

    static_dir = Path(__file__).absolute().parent.parent / "static"

    @app.get("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.get("/static/<path:filename>")
    def static(filename):
        return send_from_directory(static_dir, filename)

    @api.get("/health")
    def health():
        return jsonify(ok=True, service="nightseries")

    register_series(api, service)
    register_exposures(api, service)
    register_regions(api, service)
    register_alignment(api, service)
    register_checks(api, service)
    register_analysis(api, service)
    register_integrations(api, service)
    register_tracklet(api, service)
    from .field_api import register_field_tools
    register_field_tools(api, service)
    from .analysis_api import register_inspection_tools
    register_inspection_tools(api, service)
    app.register_blueprint(api)
    return app


def handle_http_error(exc):
    if request.path.startswith("/api/"):
        message = "上传超过工作台容量" if exc.code == 413 else exc.description
        return jsonify(error=message), exc.code
    return exc


def register_series(api, service):
    store = service.store

    @api.get("/series")
    def series_list():
        return jsonify(items=store.list())

    @api.post("/series")
    def create_series():
        return jsonify(store.create(request_data().get("name"))), 201

    @api.post("/demo/<scenario>")
    def create_demo(scenario):
        enum(scenario, "合成样例", set(SCENARIOS))
        return jsonify(service.compute(lambda: make_demo(scenario, store))), 201

    @api.get("/series/<sid>")
    def get_series(sid):
        return jsonify(store.get(sid))

    @api.patch("/series/<sid>")
    def change_series(sid):
        body = request_data()
        return jsonify(store.update_settings(sid, version_for(body), body))

    @api.delete("/series/<sid>")
    def remove_series(sid):
        store.remove(sid, query_version())
        return "", 204

    @api.get("/series/<sid>/bundle")
    def get_bundle(sid):
        content = service.compute(lambda: write_bundle(store, sid))
        return attachment(content, "application/zip", f"nightseries-{sid}.zip")

    @api.post("/series/import")
    def unpack_bundle():
        offered = request.files.get("bundle")
        if offered is None:
            raise DomainError("请选择观测 ZIP")
        data = offered.stream.read(ARCHIVE_LIMIT + 1)
        result = service.compute(lambda: read_bundle(store, data))
        return jsonify(result), 201


def upload_exposures(request, store):
    import json

    files = request.files.getlist("files")
    if not 1 <= len(files) <= MAX_FRAMES:
        raise DomainError("请选择 1 到 48 张可读取的曝光")
    kind = enum(request.form.get("kind", "science"), "影像用途", ASSET_KINDS)
    common = {
        field: request.form.get(field, "")
        for field in ("started", "exposure", "band", "notes")
    }
    custom = request.form.get("rows")
    if custom:
        try:
            custom = list_value(json.loads(custom), "逐帧元数据", MAX_FRAMES)
        except (ValueError, RecursionError) as exc:
            raise DomainError("逐帧元数据不是有效 JSON") from exc
        if len(custom) != len(files):
            raise DomainError("逐帧元数据数量与原图不一致")
    else:
        custom = [{} for _ in files]
    prepared = []
    total = 0
    for file, prescribed in zip(files, custom):
        body = file.stream.read(MAX_ASSET_BYTES + 1)
        total += len(body)
        if total > MAX_SERIES_BYTES:
            raise DomainError("这批上传超过 96 MiB 原图容量")
        metadata = {**common, **object_value(prescribed, "逐帧元数据")}
        prepared.append(store.prepare_asset(file.filename, body, kind, metadata))
    return prepared


def register_exposures(api, service):
    store = service.store

    @api.post("/series/<sid>/exposures")
    def add_exposures(sid):
        version = version_for(request.form)
        result = service.compute(
            lambda: store.append_assets(sid, version, upload_exposures(request, store))
        )
        return jsonify(result), 201

    @api.patch("/series/<sid>/exposures/<frame>")
    def edit_exposure(sid, frame):
        body = request_data()
        return jsonify(store.update_exposure(sid, version_for(body), frame, body))

    @api.delete("/series/<sid>/exposures/<frame>")
    def remove_exposure(sid, frame):
        return jsonify(store.delete_exposure(sid, query_version(), frame))

    @api.get("/series/<sid>/exposures/<frame>/original")
    def download_original(sid, frame):
        row, body = store.original(sid, frame)
        media = "application/fits" if row["format"] == "FITS" else mimetypes.guess_type(row["filename"])[0]
        return attachment(body, media or "application/octet-stream", row["filename"])

    @api.get("/series/<sid>/exposures/<frame>/image.png")
    def preview_image(sid, frame):
        def render():
            images, plane = view_plane(service, sid, frame, request.args)
            saved = images.document["stretch"]
            preview, bounds = render_plane(plane, saved, show_flags=javascript_option(request.args.get("flags")))
            return image_response(preview, X_Display_Low=bounds[0], X_Display_High=bounds[1])
        return service.compute(render)

    @api.get("/series/<sid>/exposures/<frame>/inspect")
    def detector_inspect(sid, frame):
        def inspect():
            _, plane = view_plane(service, sid, frame, request.args)
            sample = inspect_pixel(plane, request.args.get("x"), request.args.get("y"))
            return jsonify(sample)
        return service.compute(inspect)

    @api.get("/series/<sid>/exposures/<frame>/histogram")
    def detector_histogram(sid, frame):
        def histogram():
            _, plane = view_plane(service, sid, frame, request.args)
            return jsonify(raster_histogram(plane))
        return service.compute(histogram)

    @api.get("/series/<sid>/exposures/<frame>/profile")
    def detector_profile(sid, frame):
        def profile():
            _, plane = view_plane(service, sid, frame, request.args)
            result = sample_line(plane, *(request.args.get(key) for key in ("x0", "y0", "x1", "y1")))
            if request.args.get("format") == "csv":
                return attachment(profile_csv(result), "text/csv; charset=utf-8", "pixel-profile.csv")
            return jsonify(result)
        return service.compute(profile)

    @api.get("/series/<sid>/exposures/<frame>/summary")
    def exposure_summary(sid, frame):
        def summary():
            images = service.images(sid)
            image = images.image(frame)
            metadata = dict(image.metadata)
            return jsonify(
                raw_statistics=image_summary(image.samples), format=image.kind,
                width=image.dimensions.width, height=image.dimensions.height, **metadata,
            )
        return service.compute(summary)


def register_regions(api, service):
    store = service.store

    @api.post("/series/<sid>/regions/<kind>")
    def create_region(sid, kind):
        body = request_data()
        return jsonify(store.change_region(sid, version_for(body), kind, "add", body)), 201

    @api.patch("/series/<sid>/regions/<kind>/<rid>")
    def update_region(sid, kind, rid):
        body = request_data()
        return jsonify(store.change_region(sid, version_for(body), kind, "edit", body, rid))

    @api.delete("/series/<sid>/regions/<kind>/<rid>")
    def delete_region(sid, kind, rid):
        return jsonify(store.change_region(sid, query_version(), kind, "delete", entry_id=rid))


def register_alignment(api, service):
    store = service.store

    @api.put("/series/<sid>/reference")
    def make_reference(sid):
        body = request_data()
        return jsonify(store.set_reference(sid, version_for(body), body.get("frame")))

    @api.put("/series/<sid>/align/<frame>")
    def set_alignment(sid, frame):
        body = request_data()
        return jsonify(store.update_alignment(sid, version_for(body), frame, body))

    @api.delete("/series/<sid>/align/<frame>")
    def clear_alignment(sid, frame):
        return jsonify(store.update_alignment(sid, query_version(), frame, None))

    @api.post("/series/<sid>/register")
    def automatic_registration(sid):
        options = request_data()

        def register():
            images = service.images(sid)
            verify_computed(images.document, options)
            report, accepted = run_registration(images, options)
            if accepted:
                document = store.install_alignments(sid, images.document["revision"], accepted)
            else:
                document = images.document
            report.update(kind="registration", revision=document["revision"])
            return jsonify(result=report, document=document)
        return service.compute(register)

    @api.get("/series/<sid>/difference/<a>/<b>")
    def compare_aligned(sid, a, b):
        def compare():
            images = service.images(sid)
            verify_computed(images.document, request.args)
            norm = enum(request.args.get("normalize", "median"), "差分零位", {"median", "none"})
            summary = difference_preview(preview_stream(images, a, "aligned"), preview_stream(images, b, "aligned"), norm)
            return image_response(
                summary["image"], X_Difference_Offset=summary["offset"],
                X_Difference_Range=summary["limit"], X_Difference_Valid=summary["valid"],
            )
        return service.compute(compare)


def register_checks(api, service):
    @api.get("/series/<sid>/calibration")
    def calibration_description(sid):
        def compute():
            images = service.images(sid)
            return jsonify(
                inputs=calibration_status(images.document), masters=images.master_panel(),
            )
        return service.compute(compute)

    @api.get("/series/<sid>/exposures/<frame>/stars")
    def exposure_stars(sid, frame):
        def compute():
            _, plane = view_plane(service, sid, frame, request.args)
            return jsonify(frame_quality(plane, request.args))
        return service.compute(compute)


def compute_result(service, sid, kind, options):
    images = service.images(sid)
    verify_computed(images.document, options)
    evaluator = {"photometry": light_curve, "quality": quality_report}[kind]
    return evaluator(images, options)


def output_result(kind, result):
    if kind == "photometry":
        return attachment(photometry_csv(result), "text/csv; charset=utf-8", "light-curve.csv")
    if kind == "quality":
        return attachment(quality_csv(result), "text/csv; charset=utf-8", "exposure-quality.csv")
    return jsonify(result)


def register_analysis(api, service):
    store = service.store

    @api.post("/series/<sid>/analysis/<kind>")
    def calculate(sid, kind):
        enum(kind, "计算类别", {"photometry", "quality"})
        options = request_data()

        def compute():
            result = compute_result(service, sid, kind, options)
            return output_result(kind, result) if options.get("format") == "csv" else jsonify(result)
        return service.compute(compute)

    @api.get("/series/<sid>/snapshots")
    def archives(sid):
        return jsonify(items=store.list_snapshots(sid))

    @api.post("/series/<sid>/snapshots")
    def save_snapshot(sid):
        body = request_data()
        kind = enum(body.get("kind"), "归档类型", {"photometry", "quality"})
        options = object_value(body.get("options"), "计算选项")
        version = version_for(body)
        label = text(body.get("label"), "归档题名", 70)

        def compute():
            result = compute_result(service, sid, kind, {**options, "version": version})
            return jsonify(store.save_snapshot(sid, version, label, kind, result)), 201
        return service.compute(compute)

    @api.get("/series/<sid>/snapshots/<snapshot>")
    def inspect_snapshot(sid, snapshot):
        item = store.get_snapshot(sid, snapshot)
        if request.args.get("format") == "csv":
            return output_result(item["kind"], item["result"])
        return jsonify(item)

    @api.delete("/series/<sid>/snapshots/<snapshot>")
    def delete_snapshot(sid, snapshot):
        store.delete_snapshot(sid, snapshot)
        return "", 204


def register_tracklet(api, service):
    @api.post("/series/<sid>/tracklet")
    def moving_tracklet(sid):
        options = request_data()

        def compute():
            images = service.images(sid)
            verify_computed(images.document, options)
            result = tracklet_report(images, options)
            if options.get("format") == "csv":
                return attachment(tracklet_csv(result), "text/csv; charset=utf-8", "moving-tracklet.csv")
            return jsonify(result)
        return service.compute(compute)


def register_integrations(api, service):
    @api.post("/series/<sid>/stack")
    def stack_report(sid):
        options = request_data()

        def compute():
            images = service.images(sid)
            verify_computed(images.document, options)
            product = compute_stack(images, options)
            mode = enum(options.get("format", "json"), "叠加文件", {"json", "image", "coverage", "fits", "coverage_fits"})
            if mode == "image":
                rendered, _ = render_plane(product.plane, images.document["stretch"])
                return image_response(rendered)
            if mode == "coverage":
                return image_response(coverage_preview(product.coverage, len(product.report["used"])))
            if mode == "fits":
                body = make_fits(product.plane.values, {"BUNIT": "ADU/s", "FILTER": product.report["band"]})
                return attachment(body, "application/fits", "aligned-stack.fits")
            if mode == "coverage_fits":
                body = make_fits(product.coverage, {"BUNIT": "frames"}, unsigned=True)
                return attachment(body, "application/fits", "stack-support.fits")
            return jsonify(product.report)
        return service.compute(compute)
