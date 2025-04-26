"""
Tool parser for the “New-Model” chat template that emits either:

  {"name": "...", "parameters": {...}}<|eot_id|>
or
  <|python_tag|>tool_name.call(arg="…")<|eom_id|>

Exactly one tool call is produced per assistant turn (enforced by the template).
"""

from __future__ import annotations

import json
import re
from typing import Sequence, Dict, Any

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ToolCall,
    ExtractedToolCallInformation,
)
from vllm.entrypoints.openai.tool_parsers.base import (
    ToolParser,
    ToolParserManager,
)

# ──────────────────────────────────────────────────────────────────────────
@ToolParserManager.register_module(["new_model_json"])
class NewModelToolParser(ToolParser):
    """Parses tool calls emitted by the New-Model chat template."""

    _START_ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>"
    _EOT            = "<|eot_id|>"
    _EOM            = "<|eom_id|>"
    _PYTHON_TAG     = "<|python_tag|>"

    # JSON tool-call pattern
    _JSON_RE = re.compile(
        r'\{"name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*"parameters"\s*:\s*(?P<params>\{.*?\})\}',
        re.DOTALL,
    )
    # ipython tool-call pattern
    _PY_RE = re.compile(
        r'<\|python_tag\|>(?P<name>\w+)\.call\((?P<args>.*?)\)',
        re.DOTALL,
    )

    # ------------------------------------------------------------------ #
    # Request tweaks – we must **not** strip special tokens, otherwise
    # the <|eot_id|> / <|python_tag|> markers would disappear.
    # ------------------------------------------------------------------ #
    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request.skip_special_tokens = False
        return request

    # ------------------------------------------------------------------ #
    # Non-streaming parse (entire completion ready).
    # ------------------------------------------------------------------ #
    def extract_tool_calls(  # type: ignore[override]
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        tools: list[ToolCall] = []

        # 1. Look for the JSON form
        if (m := self._JSON_RE.search(model_output)):
            name     = m.group("name")
            params_s = m.group("params")
            tools.append(ToolCall(name=name,
                                  arguments=json.loads(params_s)))

        # 2. Otherwise look for the ipython form
        elif (m := self._PY_RE.search(model_output)):
            name      = m.group("name")
            args_s    = m.group("args")
            kwargs: Dict[str, Any] = {}
            if args_s.strip():
                for pair in args_s.split(","):
                    k, v = pair.split("=", 1)
                    kwargs[k.strip()] = json.loads(v.strip()) \
                        if v.strip().startswith(("{", "[")) \
                        else v.strip().strip("\"'")
            tools.append(ToolCall(name=name, arguments=kwargs))

        # 3. Return results
        if tools:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tools,
                content=None,           # no extra assistant text
            )

        # No tool call found – treat whole output as assistant content
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output,
        )

    # ------------------------------------------------------------------ #
    # Streaming parse – fire **once** when we see <|eot_id|> or <|eom_id|>
    # ------------------------------------------------------------------ #
    def extract_tool_calls_streaming(  # type: ignore[override]
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Wait until the tool call is closed
        if self._EOT not in current_text and self._EOM not in current_text:
            return None

        info = self.extract_tool_calls(current_text, request)
        if not info.tools_called:
            return None  # shouldn’t happen, but be safe

        return DeltaMessage(role="assistant",
                            content=None,
                            tool_calls=info.tool_calls)
