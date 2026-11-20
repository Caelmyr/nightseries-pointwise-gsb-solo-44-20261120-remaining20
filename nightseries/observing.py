"""Review real shutter intervals and apply independently exported observing logs."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from .domain import DomainError, MAX_FRAMES, enum, iso_time, number, science_metadata, text, usable_science_frames
from .sheets import read_sheet

LOG_COLUMNS = ("filename", "started_utc", "exposure_seconds", "filter", "notes", "enabled")


@dataclass(frozen=True)
class ExposureInterval:
    frame_id: str
    name: str
    band: str
    start: datetime
    exposure: float

    @classmethod
    def from_metadata(cls, row):
        start = parse_utc(row["started"])
        interval = cls(row["id"], row["filename"], row["band"], start, row["exposure"])
        try:
            interval.finish
        except OverflowError as exc:
            raise DomainError(f"{row['filename']} 的结束时间超出日历范围") from exc
        return interval

    @property
    def finish(self):
        return self.start + timedelta(seconds=self.exposure)

    def describe(self):
        return {
            "frame": self.frame_id,
            "filename": self.name,
            "band": self.band,
            "started": display_utc(self.start),
            "ended": display_utc(self.finish),
            "exposure": self.exposure,
        }


def parse_utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def display_utc(date):
    return date.isoformat(timespec="microseconds").replace("+00:00", "Z")


def cadence_by_filter(intervals):
    by_band = defaultdict(list)
    for frame in intervals:
        by_band[frame.band].append(frame)
    rows = []
    for band, entries in sorted(by_band.items()):
        cadence = [(later.start - earlier.start).total_seconds() for earlier, later in zip(entries, entries[1:])]
        cadence.sort()
        median = None
        length = len(cadence)
        if length:
            middle = length // 2
            median = cadence[middle] if length % 2 else (cadence[middle] + cadence[middle - 1]) * 0.5
        rows.append({
            "band": band,
            "count": len(entries),
            "exposure_seconds": round(math.fsum(entry.exposure for entry in entries), 6),
            "median_cadence": median,
            "first": display_utc(entries[0].start),
            "last": display_utc(max(entry.finish for entry in entries)),
        })
    return rows


def merge_intervals(intervals, gap_limit):
    if not intervals:
        return [], [], [], 0.0
    left = intervals[0].start
    right = intervals[0].finish
    current = [intervals[0].frame_id]
    chunks = []
    gaps = []
    overlaps = []
    seconds_covered = 0.0
    for interval in intervals[1:]:
        start, finish = interval.start, interval.finish
        if start > right:
            duration = (right - left).total_seconds()
            seconds_covered += duration
            chunks.append({"start": display_utc(left), "end": display_utc(right), "frames": current, "seconds": duration})
            gap = (start - right).total_seconds()
            gaps.append({
                "from": display_utc(right), "to": display_utc(start),
                "seconds": gap, "large": gap >= gap_limit, "before_frame": interval.frame_id,
            })
            left, right, current = start, finish, [interval.frame_id]
        else:
            if start < right:
                overlap = (min(finish, right) - start).total_seconds()
                overlaps.append({"frame": interval.frame_id, "seconds": overlap, "with": current[-1]})
            right = max(finish, right)
            current.append(interval.frame_id)
    duration = (right - left).total_seconds()
    seconds_covered += duration
    chunks.append({"start": display_utc(left), "end": display_utc(right), "frames": current, "seconds": duration})
    return chunks, gaps, overlaps, seconds_covered


def coverage_report(doc, options):
    gap_limit = number(options.get("gap_limit", 120), "长间隔提示秒数", 1, 86400)
    band = options.get("band")
    if band is not None:
        band = text(band, "滤镜", 24, required=False)
    selected = usable_science_frames(doc, band)
    if not selected:
        raise DomainError("没有符合所选滤镜的参与曝光")
    intervals = [ExposureInterval.from_metadata(row) for row in selected]
    chunks, gaps, overlaps, coverage = merge_intervals(intervals, gap_limit)
    start = intervals[0].start
    end = max(frame.finish for frame in intervals)
    times = (end - start).total_seconds()
    return {
        "revision": doc["revision"],
        "band": band,
        "gap_limit": gap_limit,
        "started": display_utc(start),
        "ended": display_utc(end),
        "span_seconds": times,
        "covered_seconds": coverage,
        "duty_cycle": coverage / times if times else 0,
        "by_band": cadence_by_filter(intervals),
        "chunks": chunks,
        "frames": [frame.describe() for frame in intervals],
        "gaps": gaps,
        "overlaps": overlaps,
    }


def explicit_utc(value, row_number):
    if not value:
        return None
    raw = text(value, "日志时间", 38)
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DomainError(f"日志第 {row_number} 行的日期时间无效") from exc
    if dt.utcoffset() is None:
        raise DomainError(f"日志第 {row_number} 行需明确写 Z 或时区偏移")
    return iso_time(raw)


def draft_log_patch(doc, log_text):
    rows = read_sheet(log_text, LOG_COLUMNS, {"filename"}, MAX_FRAMES)
    by_filename = defaultdict(list)
    for frame in doc["frames"]:
        by_filename[frame["filename"]].append(frame)
    seen_counts = Counter(row["filename"] for _, row in rows)
    accepted = []
    issues = []
    for line, row in rows:
        filename = row["filename"]
        candidates = by_filename.get(filename, [])
        base = {"line": line, "filename": filename}
        if seen_counts[filename] > 1:
            issues.append({**base, "reason": "日志文件名出现多次"})
            continue
        if len(candidates) != 1:
            reason = "观测组内该文件名不唯一" if candidates else "观测组中没有此文件"
            issues.append({**base, "reason": reason})
            continue
        old = candidates[0]
        try:
            update = row_patch(row, line)
            revised = science_metadata({**old, **update})
            ExposureInterval.from_metadata({**old, **revised})
        except DomainError as exc:
            issues.append({**base, "reason": str(exc)})
            continue
        diff = {name: value for name, value in revised.items() if old[name] != value}
        before = {name: old[name] for name in diff}
        accepted.append({**base, "frame": old["id"], "before": before, "after": diff})
    return {"revision": doc["revision"], "updates": accepted, "issues": issues}


def row_patch(row, line):
    update = {}
    if row.get("started_utc"):
        update["started"] = explicit_utc(row["started_utc"], line)
    if row.get("exposure_seconds"):
        update["exposure"] = row["exposure_seconds"]
    if row.get("filter"):
        update["band"] = "" if row["filter"] == "{none}" else row["filter"]
    if row.get("notes"):
        update["notes"] = "" if row["notes"] == "{none}" else row["notes"]
    if row.get("enabled"):
        value = enum(row["enabled"].lower(), "日志参与状态", {"true", "false", "1", "0"})
        update["enabled"] = value in {"true", "1"}
    return update
