"""Bounded CSV exchange for observer-supplied records."""
import csv
from io import StringIO

from .domain import DomainError

MAX_SHEET_BYTES = 256 * 1024


def read_sheet(body, allowed, required, row_limit):
    if isinstance(body, bytes):
        if len(body) > MAX_SHEET_BYTES:
            raise DomainError("CSV 表格超过 256 KiB")
        try:
            body = body.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DomainError("CSV 使用 UTF-8 编码") from exc
    if not isinstance(body, str) or len(body.encode("utf-8")) > MAX_SHEET_BYTES:
        raise DomainError("CSV 表格超过 256 KiB")
    io = StringIO(body, newline="")
    try:
        reader = csv.reader(io, strict=True)
        headers = next(reader, [])
        headers = [name.strip().lstrip("\ufeff") for name in headers]
        if len(headers) != len(set(headers)):
            raise DomainError("CSV 含重复列名")
        if not set(required).issubset(headers):
            raise DomainError("CSV 缺少列：" + "、".join(sorted(set(required) - set(headers))))
        if set(headers) - set(allowed):
            raise DomainError("CSV 含无法识别的列：" + "、".join(sorted(set(headers) - set(allowed))))
        rows = []
        for values in reader:
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) != len(headers):
                raise DomainError(f"CSV 第 {reader.line_num} 行和表头列数不同")
            if len(rows) >= row_limit:
                raise DomainError(f"CSV 最多 {row_limit} 个有效数据行")
            rows.append((reader.line_num, dict(zip(headers, map(str.strip, values)))))
        return rows
    except csv.Error as exc:
        raise DomainError(f"无法解析 CSV：{exc}") from exc
