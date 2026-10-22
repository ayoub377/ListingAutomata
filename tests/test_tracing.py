"""Tests for product_catalog.tracing — redaction and run labelling, nothing sent anywhere."""

from __future__ import annotations

import base64

import pytest
from langchain_core.messages import HumanMessage

from product_catalog import tracing


@pytest.fixture(autouse=True)
def tracing_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)


def _data_url(size: int) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(b"x" * size).decode()


def test_redact_images_replaces_a_data_url_with_its_size() -> None:
    assert tracing.redact_images(_data_url(300 * 1024)) == "<image 300 KB>"


def test_redact_images_reaches_nested_request_payloads() -> None:
    payload = {
        "messages": [
            {"content": [
                {"type": "text", "text": "Title: Chaise"},
                {"type": "image_url", "image_url": {"url": _data_url(2048)}},
            ]}
        ]
    }

    redacted = tracing.redact_images(payload)

    parts = redacted["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "Title: Chaise"}
    assert parts[1]["image_url"]["url"] == "<image 2 KB>"


def test_redact_images_handles_langchain_messages() -> None:
    message = HumanMessage(content=[{"type": "image_url", "image_url": {"url": _data_url(4096)}}])

    assert "<image 4 KB>" in str(tracing.redact_images(message))
    assert "base64" not in str(tracing.redact_images(message))


def test_redact_images_leaves_ordinary_values_alone() -> None:
    assert tracing.redact_images({"n": 3, "ok": True, "s": "https://img/a.jpg", "none": None}) == {
        "n": 3, "ok": True, "s": "https://img/a.jpg", "none": None,
    }


def test_run_config_names_tags_and_groups_the_trace() -> None:
    config = tracing.run_config({"name": "chaise", "title": "Chaise Camping", "shots": ["angle", "context"]}, "b1")

    assert config["run_name"] == "product:chaise"
    assert config["tags"] == ["product-catalog"]
    assert config["metadata"] == {
        "product": "chaise", "title": "Chaise Camping", "shots": ["angle", "context"], "batch_id": "b1",
    }


def test_run_config_adds_no_tracer_when_tracing_is_off() -> None:
    """Nothing may reach the network, and no Client may be built, with tracing disabled."""
    assert "callbacks" not in tracing.run_config({"name": "chaise"}, "b1")


def test_batch_ids_are_unique() -> None:
    assert tracing.new_batch_id() != tracing.new_batch_id()


def test_flush_is_a_no_op_when_tracing_is_off() -> None:
    tracing.flush()  # would raise or hang trying to reach LangSmith if it were not
