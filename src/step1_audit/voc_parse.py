from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def parse_voc_xml(path: Path, sample_id: str, allowed_classes: set[str]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return {"xml_filename": None, "xml_size": None, "objects": []}, [{
            "severity": "fatal", "sample_id": sample_id, "code": "xml_parse_error", "message": str(exc)
        }]

    filename = root.findtext("filename")
    size_node = root.find("size")
    size = None
    if size_node is not None:
        try:
            size = {
                "width": int(size_node.findtext("width", "0")),
                "height": int(size_node.findtext("height", "0")),
                "depth": int(size_node.findtext("depth", "0")),
            }
        except ValueError as exc:
            issues.append({"severity": "fatal", "sample_id": sample_id, "code": "invalid_xml_size", "message": str(exc)})

    expected_filename = f"{sample_id}.jpg"
    if filename != expected_filename:
        issues.append({
            "severity": "warning", "sample_id": sample_id, "code": "xml_filename_mismatch",
            "message": f"XML filename={filename!r}, actual={expected_filename!r}",
        })

    objects: list[dict[str, Any]] = []
    for index, node in enumerate(root.findall("object")):
        class_name = (node.findtext("name") or "").strip()
        if class_name not in allowed_classes:
            issues.append({"severity": "fatal", "sample_id": sample_id, "code": "unknown_class", "message": class_name})
        box_node = node.find("bndbox")
        try:
            if box_node is None:
                raise ValueError("missing bndbox")
            box = [int(float(box_node.findtext(key, "nan"))) for key in ("xmin", "ymin", "xmax", "ymax")]
        except ValueError as exc:
            issues.append({"severity": "fatal", "sample_id": sample_id, "code": "invalid_bbox", "message": f"object {index}: {exc}"})
            continue
        xmin, ymin, xmax, ymax = box
        valid = xmin >= 0 and ymin >= 0 and xmax > xmin and ymax > ymin
        if size:
            valid = valid and xmax <= size["width"] and ymax <= size["height"]
        if not valid:
            issues.append({"severity": "fatal", "sample_id": sample_id, "code": "invalid_bbox", "message": f"object {index}: {box}"})
        objects.append({
            "object_index": index,
            "class_name": class_name,
            "bbox_xyxy": box,
            "width": xmax - xmin,
            "height": ymax - ymin,
            "area": max(0, xmax - xmin) * max(0, ymax - ymin),
            "difficult_raw": int(node.findtext("difficult", "0")),
            "truncated_raw": int(node.findtext("truncated", "0")),
            "pose": node.findtext("pose", "Unspecified"),
        })
    if not objects:
        issues.append({"severity": "warning", "sample_id": sample_id, "code": "empty_annotation", "message": "No valid object records"})
    return {"xml_filename": filename, "xml_size": size, "objects": objects}, issues
