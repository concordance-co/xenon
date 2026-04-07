from pipelines.interp.tool_schemas import (
    TRADING_DECISION_TOOLS_V1,
    build_structured_tool_call_schema,
)


def test_build_structured_tool_call_schema_for_required_choice():
    schema = build_structured_tool_call_schema(TRADING_DECISION_TOOLS_V1, "required")

    assert schema is not None
    assert schema["type"] == "object"
    assert len(schema["anyOf"]) == 3
    tool_names = {
        option["properties"]["name"]["enum"][0]
        for option in schema["anyOf"]
    }
    assert tool_names == {"buy_token", "sell_token", "record_observation"}


def test_build_structured_tool_call_schema_for_named_tool():
    schema = build_structured_tool_call_schema(
        TRADING_DECISION_TOOLS_V1,
        {"type": "function", "function": {"name": "record_observation"}},
    )

    assert schema is not None
    assert schema["properties"]["name"]["enum"] == ["record_observation"]
    assert schema["required"] == ["name", "arguments"]
