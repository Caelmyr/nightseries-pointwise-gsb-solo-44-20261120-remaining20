"""Durable observation state and immutable original exposure blobs."""
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from secrets import token_hex
import json
import sqlite3

from .domain import (
    ASSET_KINDS, MAX_APERTURES, MAX_ASSET_BYTES, MAX_FRAMES, MAX_MASKS, MAX_SERIES,
    MAX_SERIES_BYTES, MAX_SNAPSHOTS, DomainError, alignment_value, aperture_value,
    calibration_metadata, calibration_settings, dimensions_of, enum, exposure_row,
    identifier, integer, mask_value, new_document, object_value, photometry_settings,
    science_metadata, stretch_value, text,
)
from .raster import read_image

SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    id TEXT PRIMARY KEY,
    updated TEXT NOT NULL,
    revision INTEGER NOT NULL,
    document TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS asset (
    series_id TEXT NOT NULL REFERENCES series(id) ON DELETE CASCADE,
    id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    body BLOB NOT NULL,
    PRIMARY KEY(series_id, id)
);
CREATE TABLE IF NOT EXISTS snapshot (
    series_id TEXT NOT NULL REFERENCES series(id) ON DELETE CASCADE,
    id TEXT NOT NULL,
    created TEXT NOT NULL,
    revision INTEGER NOT NULL,
    label TEXT NOT NULL,
    kind TEXT NOT NULL,
    result TEXT NOT NULL,
    PRIMARY KEY(series_id, id)
);
CREATE INDEX IF NOT EXISTS snapshot_time ON snapshot(series_id, created DESC);
"""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def new_id():
    return token_hex(8)


class Store:

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript(SCHEMA)
        finally:
            db.close()

    @contextmanager
    def session(self, write=False):
        db = sqlite3.connect(self.path, isolation_level=None, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except sqlite3.OperationalError as exc:
            db.rollback()
            if "locked" in str(exc):
                raise DomainError("观测库正在处理另一次写入，请刷新后再试", 503) from exc
            raise
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _row(db, series_id):
        identifier(series_id, "观测组")
        row = db.execute("SELECT * FROM series WHERE id=?", (series_id,)).fetchone()
        if row is None:
            raise DomainError("观测组不存在", 404)
        document = json.loads(row["document"])
        document.update(id=row["id"], revision=row["revision"], updated=row["updated"])
        return document

    @staticmethod
    def _require_revision(document, version):
        expected = integer(version, "观测版本", 1, 2**52)
        if document["revision"] != expected:
            raise DomainError("观测组已发生变动，请刷新后重做当前操作", 409)

    @staticmethod
    def _edit(db, document):
        record = {key: value for key, value in document.items() if key not in {"id", "updated", "revision"}}
        document["revision"] += 1
        document["updated"] = utc_now()
        db.execute(
            "UPDATE series SET revision=?,updated=?,document=? WHERE id=?",
            (document["revision"], document["updated"], encode(record), document["id"]),
        )
        return document

    @staticmethod
    def _create(db, document):
        count = db.execute("SELECT count(*) FROM series").fetchone()[0]
        if count >= MAX_SERIES:
            raise DomainError(f"本机最多保存 {MAX_SERIES} 组观测")
        series_id = new_id()
        updated = utc_now()
        db.execute("INSERT INTO series VALUES (?,?,?,?)", (series_id, updated, 1, encode(document)))
        return {**document, "id": series_id, "updated": updated, "revision": 1}

    def list(self):
        with self.session() as db:
            results = db.execute(
                "SELECT id, updated, revision, document FROM series ORDER BY updated DESC, id"
            ).fetchall()
        summary = []
        for row in results:
            document = json.loads(row["document"])
            frames = document["frames"]
            summary.append({
                "id": row["id"],
                "revision": row["revision"],
                "updated": row["updated"],
                "name": document["name"],
                "synthetic": document["synthetic"],
                "frames": len(frames),
                "filters": sorted({frame["band"] for frame in frames if frame["band"]}),
                "start": min((frame["started"] for frame in frames), default=None),
                "end": max((frame["started"] for frame in frames), default=None),
            })
        return summary

    def create(self, name):
        with self.session(write=True) as db:
            return self._create(db, new_document(name))

    def get(self, series_id):
        with self.session() as db:
            return self._row(db, series_id)

    def remove(self, series_id, version):
        with self.session(write=True) as db:
            document = self._row(db, series_id)
            self._require_revision(document, version)
            db.execute("DELETE FROM series WHERE id=?", (series_id,))

    def update_settings(self, series_id, version, data):
        data = object_value(data, "设置")
        with self.session(write=True) as db:
            doc = self._row(db, series_id)
            self._require_revision(doc, version)
            if "name" in data:
                doc["name"] = text(data["name"], "观测组名称", 80)
            if "notes" in data:
                doc["notes"] = text(data["notes"], "观测备注", 350, required=False)
            for key, check in (
                ("photometry", photometry_settings),
                ("calibration", calibration_settings),
                ("stretch", stretch_value),
            ):
                if key in data:
                    proposed = object_value(data[key], "设置")
                    doc[key] = check({**doc[key], **proposed})
            return self._edit(db, doc)

    def set_reference(self, series_id, version, frame_id):
        with self.session(write=True) as db:
            doc = self._row(db, series_id)
            self._require_revision(doc, version)
            exposure_row(doc, frame_id)
            if frame_id == doc["reference"]:
                return doc
            if doc["apertures"]:
                raise DomainError("已有基准坐标孔径。切换基准前请先移除孔径。", 409)
            doc["reference"] = frame_id
            doc["alignments"] = {}
            doc["plate"] = None
            return self._edit(db, doc)

    @staticmethod
    def prepare_asset(filename, body, kind, data):
        enum(kind, "影像用途", ASSET_KINDS)
        name = text(Path(filename).name, "文件名", 140)
        image = read_image(name, body)
        metadata = (
            science_metadata(data, image.metadata) if kind == "science"
            else calibration_metadata(kind, data, image.metadata)
        )
        asset_row = {
            "id": new_id(),
            "filename": name,
            "kind": kind,
            "format": image.kind,
            "sha256": sha256(body).hexdigest(),
            "byte_size": len(body),
            "header": image.metadata.get("header", {}),
            **metadata,
        }
        return asset_row, image.dimensions, body

    @staticmethod
    def _apply_assets(db, doc, prepared):
        frames = doc["frames"]
        calibration = doc["calibration_frames"]
        if len(frames) + len(calibration) + len(prepared) > MAX_FRAMES:
            raise DomainError(f"一组观测含校准帧最多 {MAX_FRAMES} 张图")
        base_size = sum(value["byte_size"] for value in frames + calibration)
        if base_size + sum(row["byte_size"] for row, _, _ in prepared) > MAX_SERIES_BYTES:
            raise DomainError("观测组超过 96 MiB 原图容量")
        for row, dimensions, body in prepared:
            actual = {"width": dimensions.width, "height": dimensions.height}
            if doc["dimensions"] is None:
                doc["dimensions"] = actual
            if doc["dimensions"] != actual:
                raise DomainError(f"{row['filename']} 的画幅与观测组不同")
            if row["kind"] == "science":
                frames.append(row)
                if doc["reference"] is None:
                    doc["reference"] = row["id"]
            else:
                calibration.append(row)
            db.execute(
                "INSERT INTO asset VALUES (?,?,?,?)",
                (doc["id"], row["id"], row["sha256"], body),
            )
        frames.sort(key=lambda value: (value["started"], value["id"]))

    def append_assets(self, series_id, version, prepared):
        with self.session(write=True) as db:
            document = self._row(db, series_id)
            self._require_revision(document, version)
            self._apply_assets(db, document, prepared)
            return self._edit(db, document)

    def create_populated(self, document, prepared, initial_state=None, snapshots=None):
        with self.session(write=True) as db:
            record = self._create(db, document)
            self._apply_assets(db, record, prepared)
            if initial_state:
                record.update(initial_state)
            for snapshot in snapshots or []:
                self._insert_snapshot(db, record, snapshot["label"], snapshot["kind"], snapshot["result"], snapshot["revision"], snapshot["created"])
            return self._edit(db, record)

    def update_exposure(self, series_id, version, frame_id, data):
        with self.session(write=True) as db:
            doc = self._row(db, series_id)
            self._require_revision(doc, version)
            frame = exposure_row(doc, frame_id, include_calibration=True)
            merged = {**frame, **object_value(data, "影像设置")}
            if frame["kind"] == "science":
                frame.update(science_metadata(merged))
                doc["frames"].sort(key=lambda entry: (entry["started"], entry["id"]))
            else:
                frame.update(calibration_metadata(frame["kind"], merged))
            return self._edit(db, doc)

    def delete_exposure(self, series_id, version, frame_id):
        with self.session(write=True) as db:
            doc = self._row(db, series_id)
            self._require_revision(doc, version)
            frame = exposure_row(doc, frame_id, include_calibration=True)
            if frame_id == doc["reference"] and doc["apertures"]:
                raise DomainError("基准帧承载现有孔径坐标，请先移除孔径", 409)
            frame_list = doc["frames"] if frame["kind"] == "science" else doc["calibration_frames"]
            frame_list.remove(frame)
            doc["alignments"].pop(frame_id, None)
            doc["masks"] = [mask for mask in doc["masks"] if mask["frame"] != frame_id]
            if doc["reference"] == frame_id:
                doc["reference"] = doc["frames"][0]["id"] if doc["frames"] else None
                doc["alignments"] = {}
                doc["plate"] = None
            if not doc["frames"] and not doc["calibration_frames"]:
                doc["dimensions"] = None
                doc["masks"] = []
            db.execute("DELETE FROM asset WHERE series_id=? AND id=?", (series_id, frame_id))
            return self._edit(db, doc)

    def original(self, series_id, frame_id):
        with self.session() as db:
            document = self._row(db, series_id)
            row = exposure_row(document, frame_id, include_calibration=True)
            asset = db.execute(
                "SELECT body, fingerprint FROM asset WHERE series_id=? AND id=?", (series_id, frame_id)
            ).fetchone()
            if asset is None or asset["fingerprint"] != row["sha256"]:
                raise DomainError("原曝光文件与观测记录不一致", 409)
            return row, asset["body"]

    def assets(self, series_id, frame_ids=None):
        with self.session() as db:
            document = self._row(db, series_id)
            if frame_ids is None:
                rows = db.execute("SELECT id, body FROM asset WHERE series_id=?", (series_id,))
                return document, {row["id"]: row["body"] for row in rows}
            chosen = list(dict.fromkeys(frame_ids))
            bodies = {}
            for frame_id in chosen:
                exposure_row(document, frame_id, include_calibration=True)
                asset = db.execute(
                    "SELECT body FROM asset WHERE series_id=? AND id=?", (series_id, frame_id)
                ).fetchone()
                if asset is None:
                    raise DomainError("曝光原图丢失", 409)
                bodies[frame_id] = asset["body"]
            return document, bodies

    def update_alignment(self, series_id, version, frame_id, data):
        with self.session(write=True) as db:
            doc = self._row(db, series_id)
            self._require_revision(doc, version)
            exposure_row(doc, frame_id)
            if frame_id == doc["reference"]:
                raise DomainError("基准帧始终位于原始坐标")
            if data is None:
                doc["alignments"].pop(frame_id, None)
            else:
                doc["alignments"][frame_id] = alignment_value(data)
            return self._edit(db, doc)

    def install_alignments(self, series_id, version, entries):
        with self.session(write=True) as db:
            doc = self._row(db, series_id)
            self._require_revision(doc, version)
            for frame_id, entry in entries.items():
                exposure_row(doc, frame_id)
                if frame_id == doc["reference"]:
                    raise DomainError("基准帧不参与偏移写入")
                doc["alignments"][frame_id] = entry
            return self._edit(db, doc)

    def change_region(self, series_id, version, kind, action, data=None, entry_id=None):
        if kind not in {"apertures", "masks"}:
            raise DomainError("未知区域类型")
        with self.session(write=True) as db:
            doc = self._row(db, series_id)
            self._require_revision(doc, version)
            dims = dimensions_of(doc)
            if kind == "apertures" and doc["reference"] is None:
                raise DomainError("请先建立科学曝光基准")
            entries = doc[kind]
            if action == "add":
                maximum = MAX_APERTURES if kind == "apertures" else MAX_MASKS
                if len(entries) >= maximum:
                    raise DomainError(f"该类区域最多保存 {maximum} 个")
                old = None
            else:
                old = next((entry for entry in entries if entry["id"] == entry_id), None)
                if not old:
                    raise DomainError("没有这个区域", 404)
                if action == "delete":
                    entries.remove(old)
                    return self._edit(db, doc)
            merged = {**(old or {}), **object_value(data, "区域")}
            if kind == "apertures":
                region = aperture_value(merged, dims)
                if any(entry is not old and entry["label"] == region["label"] for entry in entries):
                    raise DomainError("同一观测组的孔径标签须唯一")
            else:
                region = mask_value(merged, dims)
                if region["frame"]:
                    exposure_row(doc, region["frame"])
            if old is None:
                entries.append({**region, "id": new_id()})
            else:
                old.update(region)
            return self._edit(db, doc)

    def append_detector_masks(self, series_id, version, masks):
        with self.session(write=True) as db:
            document = self._row(db, series_id)
            self._require_revision(document, version)
            dims = dimensions_of(document)
            existing = document["masks"]
            keys = ("left", "top", "right", "bottom", "frame")
            saved = {tuple(mask[key] for key in keys) for mask in existing}
            added = []
            for draft in masks:
                region = mask_value(draft, dims)
                key = tuple(region[name] for name in keys)
                if key in saved:
                    continue
                saved.add(key)
                added.append({"id": new_id(), **region})
            if len(existing) + len(added) > MAX_MASKS:
                raise DomainError(f"全组最多保存 {MAX_MASKS} 个坏区")
            if added:
                existing.extend(added)
                return self._edit(db, document)
            return document

    def set_plate(self, series_id, version, plate):
        with self.session(write=True) as db:
            document = self._row(db, series_id)
            self._require_revision(document, version)
            if plate and not document["reference"]:
                raise DomainError("请先指定一张参考曝光")
            document["plate"] = {**plate, "reference": document["reference"]} if plate else None
            return self._edit(db, document)

    def set_catalog(self, series_id, version, catalog):
        with self.session(write=True) as db:
            document = self._row(db, series_id)
            self._require_revision(document, version)
            document["catalog"] = catalog
            return self._edit(db, document)

    def apply_log(self, series_id, version, updates):
        with self.session(write=True) as db:
            document = self._row(db, series_id)
            self._require_revision(document, version)
            altered = False
            for row in updates:
                if not row["after"]:
                    continue
                frame = exposure_row(document, row["frame"])
                frame.update(science_metadata({**frame, **row["after"]}))
                altered = True
            if altered:
                document["frames"].sort(key=lambda frame: (frame["started"], frame["id"]))
                return self._edit(db, document)
            return document

    @staticmethod
    def _insert_snapshot(db, doc, label, kind, result, revision=None, created=None):
        count = db.execute("SELECT count(*) FROM snapshot WHERE series_id=?", (doc["id"],)).fetchone()[0]
        if count >= MAX_SNAPSHOTS:
            raise DomainError(f"每组最多 {MAX_SNAPSHOTS} 条已保存结果")
        snap = {
            "id": new_id(),
            "created": created or utc_now(),
            "revision": doc["revision"] if revision is None else revision,
            "label": text(label, "归档题名", 70),
            "kind": enum(kind, "归档类型", {"photometry", "quality", "registration"}),
            "result": object_value(result, "归档结果"),
        }
        payload = encode(snap["result"])
        if len(payload.encode("utf-8")) > 2 * 1024 * 1024:
            raise DomainError("本次计算结果超过单个存档 2 MiB 的容量")
        db.execute(
            "INSERT INTO snapshot VALUES (?,?,?,?,?,?,?)",
            (doc["id"], snap["id"], snap["created"], snap["revision"], snap["label"], snap["kind"], payload),
        )
        return snap

    def save_snapshot(self, series_id, version, label, kind, result):
        with self.session(write=True) as db:
            doc = self._row(db, series_id)
            self._require_revision(doc, version)
            return self._insert_snapshot(db, doc, label, kind, result)

    def list_snapshots(self, series_id):
        with self.session() as db:
            self._row(db, series_id)
            rows = db.execute(
                "SELECT id,created,revision,label,kind FROM snapshot WHERE series_id=? ORDER BY created DESC,id",
                (series_id,),
            )
            return [dict(row) for row in rows]

    def get_snapshot(self, series_id, item_id):
        with self.session() as db:
            self._row(db, series_id)
            row = db.execute("SELECT * FROM snapshot WHERE series_id=? AND id=?", (series_id, item_id)).fetchone()
            if row is None:
                raise DomainError("找不到这个结果", 404)
            result = dict(row)
            result["result"] = json.loads(result["result"])
            result.pop("series_id")
            return result

    def delete_snapshot(self, series_id, item_id):
        with self.session(write=True) as db:
            self._row(db, series_id)
            result = db.execute("DELETE FROM snapshot WHERE series_id=? AND id=?", (series_id, item_id))
            if result.rowcount == 0:
                raise DomainError("找不到这个结果", 404)

    def bundle(self, series_id):
        with self.session() as db:
            doc = self._row(db, series_id)
            assets = db.execute("SELECT id,body FROM asset WHERE series_id=?", (series_id,))
            images = {row["id"]: row["body"] for row in assets}
            rows = db.execute(
                "SELECT id,created,revision,label,kind,result FROM snapshot WHERE series_id=? ORDER BY created,id",
                (series_id,),
            )
            snapshots = [dict(row) for row in rows]
            for snapshot in snapshots:
                snapshot["result"] = json.loads(snapshot["result"])
            return doc, images, snapshots
