"""Metric attributes must not carry per-request values.

opentelemetry-sdk 1.28 keeps one aggregation per unique attribute set for the
life of the meter provider and implements no cardinality limit, so a single
request-scoped attribute makes memory grow linearly with traffic forever.
This asserts the attribute sets stay constant as request count grows.
"""

import os
import sys
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig


class _Recorder:
    """Stands in for a Histogram, remembering every distinct attribute set."""

    def __init__(self):
        self.attribute_sets = set()

    def record(self, value, attributes=None):
        self.attribute_sets.add(frozenset((attributes or {}).items()))


def _make_logger():
    # Bypass __init__ so no exporter, provider or network setup is needed.
    logger = OpenTelemetry.__new__(OpenTelemetry)
    # _gen_ai_semconv_latest_experimental is a read-only property off this;
    # the default config opts into no semconv categories.
    logger.config = OpenTelemetryConfig()
    recorders = {}
    for name in (
        "_operation_duration_histogram",
        "_token_usage_histogram",
        "_cost_histogram",
        "_time_to_first_token_histogram",
        "_time_per_output_token_histogram",
        "_response_duration_histogram",
    ):
        recorders[name] = _Recorder()
        setattr(logger, name, recorders[name])
    return logger, recorders


def _make_kwargs():
    """One request's worth of kwargs, with the request-scoped fields varying."""
    return {
        "model": "bedrock/us.meta.llama3-3-70b-instruct-v1:0",
        "litellm_params": {"custom_llm_provider": "bedrock"},
        "optional_params": {"stream": False},
        "response_cost": 0.00123,
        "standard_logging_object": {
            "metadata": {
                # stable, bounded by how many keys/teams exist
                "user_api_key_hash": "d8be13191ed55305",
                "user_api_key_alias": "cigna-aidr",
                "user_api_key_team_alias": "brokers",
                "user_api_key_team_id": "054aec77-de20-4d9d",
                "applied_guardrails": ["noma-prompt-injection"],
                # varies per request - must never reach a metric attribute
                "user_api_key_end_user_id": str(uuid.uuid4()),
                "requester_ip_address": f"10.2.{uuid.uuid4().int % 255}.{uuid.uuid4().int % 255}",
                "requester_metadata": {
                    "component": "ClassifierName_IndirectPromptInjectionLLMClassifier",
                    "platform": "PLATFORM_CLAUDE_CODE",
                    "headers": {"traceparent": f"00-{uuid.uuid4().hex}-01"},
                },
                "spend_logs_metadata": {"request_id": str(uuid.uuid4())},
            },
            "hidden_params": {
                "response_cost": uuid.uuid4().int % 10**9 / 10**12,
                "litellm_overhead_time_ms": uuid.uuid4().int % 1000 / 7.0,
                "cache_key": uuid.uuid4().hex,
            },
        },
    }


def test_metric_attributes_are_bounded():
    logger, recorders = _make_logger()
    start = datetime.now()
    end = start + timedelta(seconds=1)
    response_obj = {"usage": {"prompt_tokens": 2419, "completion_tokens": 53}}

    counts = []
    for _ in range(3):
        for _ in range(200):
            logger._record_metrics(_make_kwargs(), response_obj, start, end)
        counts.append(
            sum(len(r.attribute_sets) for r in recorders.values())
        )

    assert counts[0] == counts[-1], (
        f"distinct metric attribute sets grew with request count: {counts}. "
        "A request-scoped value is leaking into metric attributes."
    )

    recorded = set()
    for recorder in recorders.values():
        for attribute_set in recorder.attribute_sets:
            recorded.update(key for key, _ in attribute_set)

    for forbidden in (
        "metadata.user_api_key_end_user_id",
        "metadata.requester_ip_address",
        "metadata.requester_metadata",
        "metadata.spend_logs_metadata",
        "hidden_params",
    ):
        assert forbidden not in recorded, f"{forbidden} must not be a metric attribute"

    # Every attribute the collector's transform/litellm_genai forwards must still
    # be emitted, under the same name, or downstream dashboards lose a dimension.
    # component/platform are unprefixed because that is what the collector
    # previously derived from requester_metadata.
    for required in (
        "gen_ai.request.model",
        "gen_ai.system",
        "metadata.user_api_key_hash",
        "metadata.user_api_key_alias",
        "metadata.user_api_key_team_id",
        "metadata.user_api_key_team_alias",
        "metadata.applied_guardrails",
        "component",
        "platform",
    ):
        assert required in recorded, f"{required} should still be a metric attribute"


if __name__ == "__main__":
    test_metric_attributes_are_bounded()
    print("OK: metric attribute cardinality is bounded")
