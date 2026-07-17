from pathlib import Path

from step1_audit.voc_parse import parse_voc_xml


def test_parses_filename_mismatch_and_object(tmp_path: Path):
    xml = tmp_path / "0001_b_test.xml"
    xml.write_text("""<annotation><filename>0001_test.jpg</filename><size><width>800</width><height>800</height><depth>3</depth></size><object><name>broke</name><pose>Unspecified</pose><truncated>1</truncated><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>20</ymax></bndbox></object></annotation>""", encoding="utf-8")
    parsed, issues = parse_voc_xml(xml, "0001_b_test", {"broke"})
    assert parsed["objects"][0]["bbox_xyxy"] == [1, 2, 10, 20]
    assert [issue["code"] for issue in issues] == ["xml_filename_mismatch"]


def test_rejects_invalid_box(tmp_path: Path):
    xml = tmp_path / "0001_test.xml"
    xml.write_text("""<annotation><filename>0001_test.jpg</filename><size><width>10</width><height>10</height><depth>3</depth></size><object><name>dirty</name><bndbox><xmin>8</xmin><ymin>2</ymin><xmax>4</xmax><ymax>20</ymax></bndbox></object></annotation>""", encoding="utf-8")
    _, issues = parse_voc_xml(xml, "0001_test", {"dirty"})
    assert any(issue["code"] == "invalid_bbox" and issue["severity"] == "fatal" for issue in issues)
