import pytest

from vieneu_utils.srt_parser import SrtBlock, SrtValidationError, parse_srt


HAPPY = """1
00:00:01,000 --> 00:00:03,000
Xin chào.

2
00:00:03,500 --> 00:00:05,000
Đây là dòng một
và dòng hai.

3
00:00:06,000 --> 00:00:07,500
Kết thúc.
"""


def test_parse_happy_path():
    blocks = parse_srt(HAPPY)
    assert len(blocks) == 3
    assert blocks[0] == SrtBlock(1, 1.0, 3.0, "Xin chào.")
    assert blocks[1].text == "Đây là dòng một\nvà dòng hai."
    assert blocks[1].start_s == pytest.approx(3.5)
    assert blocks[1].end_s == pytest.approx(5.0)
    assert blocks[2].index == 3


@pytest.mark.parametrize("raw", ["", "   \n\t\n  "])
def test_empty_input_raises(raw):
    with pytest.raises(SrtValidationError) as ei:
        parse_srt(raw)
    assert ei.value.code == "empty"


def test_malformed_timestamp_raises():
    raw = "1\n00:00:01 --> 00:00:03\nXin chào.\n"
    with pytest.raises(SrtValidationError) as ei:
        parse_srt(raw)
    assert ei.value.code == "malformed_timestamp"


def test_missing_text_raises():
    raw = "1\n00:00:01,000 --> 00:00:03,000\n"
    with pytest.raises(SrtValidationError) as ei:
        parse_srt(raw)
    assert ei.value.code == "missing_text"
    assert ei.value.block == 1


def test_blank_text_raises():
    raw = "1\n00:00:01,000 --> 00:00:03,000\n   \n"
    with pytest.raises(SrtValidationError) as ei:
        parse_srt(raw)
    assert ei.value.code == "blank_text"
    assert ei.value.block == 1


def test_out_of_order_raises():
    raw = (
        "1\n00:00:01,000 --> 00:00:02,000\nA.\n\n"
        "3\n00:00:02,500 --> 00:00:03,000\nC.\n\n"
        "2\n00:00:03,500 --> 00:00:04,000\nB.\n"
    )
    with pytest.raises(SrtValidationError) as ei:
        parse_srt(raw)
    assert ei.value.code == "bad_order"


def test_overlap_raises():
    raw = (
        "1\n00:00:01,000 --> 00:00:03,000\nA.\n\n"
        "2\n00:00:02,500 --> 00:00:04,000\nB.\n"
    )
    with pytest.raises(SrtValidationError) as ei:
        parse_srt(raw)
    assert ei.value.code == "overlap"


def test_bad_duration_raises():
    raw = "1\n00:00:03,000 --> 00:00:03,000\nA.\n"
    with pytest.raises(SrtValidationError) as ei:
        parse_srt(raw)
    assert ei.value.code == "bad_duration"
    assert ei.value.block == 1


def test_bom_and_crlf_tolerance():
    raw = "﻿1\r\n00:00:01,000 --> 00:00:02,000\r\nHello.\r\n"
    blocks = parse_srt(raw)
    assert blocks == [SrtBlock(1, 1.0, 2.0, "Hello.")]


def test_long_gap_allowed():
    raw = (
        "1\n00:00:01,000 --> 00:00:02,000\nA.\n\n"
        "2\n00:01:00,000 --> 00:01:01,000\nB.\n"
    )
    blocks = parse_srt(raw)
    assert blocks[1].start_s == pytest.approx(60.0)


def test_dot_millisecond_separator():
    raw = "1\n00:00:01.250 --> 00:00:02.500\nA.\n"
    blocks = parse_srt(raw)
    assert blocks[0].start_s == pytest.approx(1.25)
    assert blocks[0].end_s == pytest.approx(2.5)
