"""SSE parsing is forward-compatible for when the backend streams."""

from ai_gateway._sse import iter_text_chunks, parse_sse_lines


def test_parse_sse_lines_skips_done_and_comments():
    lines = [
        ": ping",
        "",
        'data: {"choices": [{"delta": {"content": "hel"}}]}',
        "",
        'data: {"choices": [{"delta": {"content": "lo"}}]}',
        "",
        "data: [DONE]",
        "",
        'data: {"choices": []}',
    ]
    events = list(parse_sse_lines(lines))
    assert len(events) == 2
    assert list(iter_text_chunks(events)) == ["hel", "lo"]
