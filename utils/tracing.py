"""Arthur tracing integration via OpenInference + OpenTelemetry.

Automatically instruments all LangChain/LangGraph calls and exports
traces to the Arthur Engine OTLP endpoint.

Required env vars:
    ARTHUR_BASE_URL   — Engine URL (e.g. http://localhost:3030)
    ARTHUR_API_KEY    — Engine API key
    ARTHUR_TASK_ID    — Agentic task/model ID (must have is_agentic=True)
"""

import logging
import os

logger = logging.getLogger(__name__)

_INSTRUMENTED = False


def setup_arthur_tracing() -> bool:
    """Initialize Arthur tracing. Returns True if tracing was enabled, False otherwise.

    Safe to call multiple times — instruments only once.
    """
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return True

    arthur_base_url = os.environ.get("ARTHUR_BASE_URL")
    arthur_api_key = os.environ.get("ARTHUR_API_KEY")
    arthur_task_id = os.environ.get("ARTHUR_TASK_ID")

    if not all([arthur_base_url, arthur_api_key, arthur_task_id]):
        logger.info(
            "Arthur tracing not configured — set ARTHUR_BASE_URL, ARTHUR_API_KEY, "
            "and ARTHUR_TASK_ID to enable."
        )
        return False

    try:
        from opentelemetry import trace as trace_api
        from opentelemetry.sdk import trace as trace_sdk
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from openinference.instrumentation.langchain import LangChainInstrumentor

        resource = Resource.create({
            "arthur.task": arthur_task_id,
            "service.name": "campaign-orchestrator-agent",
        })

        tracer_provider = trace_sdk.TracerProvider(resource=resource)
        trace_api.set_tracer_provider(tracer_provider)

        endpoint = f"{arthur_base_url.rstrip('/')}/v1/traces"
        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={"Authorization": f"Bearer {arthur_api_key}"},
        )
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

        LangChainInstrumentor().instrument()

        _INSTRUMENTED = True
        logger.info("Arthur tracing enabled — exporting to %s", endpoint)
        return True

    except Exception:
        logger.exception("Failed to initialize Arthur tracing")
        return False
