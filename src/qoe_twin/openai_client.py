"""OpenAI client used by the Digital Twin explanation layer."""

from __future__ import annotations

import os

from openai import OpenAI

from qoe_twin.explanation_schema import Explanation


class OpenAIExplanationClient:
    """Generate explanations constrained by a Pydantic schema."""

    def __init__(
        self,
        model: str | None = None,
        timeout: int = 30,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Add it to .env."
            )

        self.model = model or os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini",
        )

        self.client = OpenAI(
            api_key=api_key,
            timeout=timeout,
        )

    def generate_json(
        self,
        prompt: str,
    ) -> Explanation:
        """Generate one schema-constrained explanation."""
        response = self.client.responses.parse(
            model=self.model,
            temperature=0,
            input=prompt,
            text_format=Explanation,
        )

        explanation = response.output_parsed

        if explanation is None:
            refusal_messages: list[str] = []

            for output in response.output:
                if getattr(output, "type", None) != "message":
                    continue

                for content in getattr(
                    output,
                    "content",
                    [],
                ):
                    if getattr(
                        content,
                        "type",
                        None,
                    ) == "refusal":
                        refusal_messages.append(
                            getattr(
                                content,
                                "refusal",
                                "Model refused the request.",
                            )
                        )

            refusal_text = "; ".join(
                refusal_messages
            )

            raise ValueError(
                refusal_text
                or "OpenAI returned no parsed explanation."
            )

        return explanation
