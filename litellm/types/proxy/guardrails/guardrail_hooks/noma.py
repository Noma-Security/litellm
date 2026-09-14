from pydantic import Field

from .base import GuardrailConfigModel


class NomaGuardrailConfigModel(GuardrailConfigModel):
    use_v2: bool | None = Field(
        default=False,
        description="If True and guardrail='noma', route to the new Noma v2 implementation.",
    )
    api_key: str | None = Field(
        default=None,
        description="The Noma API key. Reads from NOMA_API_KEY env var if None.",
    )
    api_base: str | None = Field(
        default=None,
        description="The Noma API base URL. Defaults to https://api.noma.security. Also checks if the NOMA_API_KEY env var is set.",
    )
    application_id: str | None = Field(
        default=None,
        description="The Noma Application ID. Reads from NOMA_APPLICATION_ID env var if None.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Noma Security"


class NomaV2GuardrailConfigModel(GuardrailConfigModel):
    api_key: str | None = Field(
        default=None,
        description="The Noma API key. Reads from NOMA_API_KEY env var if None.",
    )
    api_base: str | None = Field(
        default=None,
        description="The Noma API base URL. Defaults to https://api.noma.security.",
    )
    application_id: str | None = Field(
        default=None,
        description="The Noma Application ID. Reads from NOMA_APPLICATION_ID env var if None.",
    )
    monitor_mode: bool | None = Field(
        default=None,
        description="When true, run guardrail checks in monitor mode.",
    )
    block_failures: bool | None = Field(
        default=None,
        description="When true, fail closed on Noma API errors.",
    )
    on_flagged_action: str | None = Field(
        default=None,
        description=(
            "What to do with a BLOCKED verdict: 'block' (HTTP 400, the default), "
            "'passthrough' (HTTP 200 whose assistant message carries the violation), "
            "or 'monitor' (log only). Reads NOMA_ON_FLAGGED_ACTION if None."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Noma Security v2"
