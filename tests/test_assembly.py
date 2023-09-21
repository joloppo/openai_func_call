from enum import Enum
from typing import Literal

from docstring_parser import parse
from pydantic import create_model

from openai_func_call.assembly import CallableFunction, func_to_callable_function


def test_func_to_callable_function():
    class EnumType(str, Enum):
        enum1: str = "enum1"
        enum2: str = "enum2"

    def test_function(
        str_param: str,
        float_param: float,
        int_param: int,
        literal_param: Literal["option1", "option2"],
        enum_param: EnumType,
        default_str_param: str = "default_str_param",
    ):
        """Description of test function here.
        :param str_param: This is the string parameter.
        :param float_param: This is the float parameter.
        :param int_param: This is the int parameter.
        :param literal_param: This is the literal parameter.
        :param enum_param: This is the enum parameter.
        :param default_str_param: This is the default string parameter.
        """
        ...

    callable_function = func_to_callable_function(test_function)
    # -- Expected --
    params_model = create_model(
        "FunctionArgsModel",
        str_param=(str, ...),
        float_param=(float, ...),
        int_param=(int, ...),
        literal_param=(Literal["option1", "option2"], ...),
        enum_param=(EnumType, ...),
        default_str_param=(str, "default_str_param"),
    )
    doc_str = parse(test_function.__doc__)
    api_dict = {
        "name": "test_function",
        "description": "Description of test function here.",
        "parameters": {
            "type": "object",
            "properties": {
                "str_param": {"type": "string", "description": "This is the string parameter."},
                "float_param": {
                    "type": "number",
                    "description": "This is the float parameter.",
                },
                "int_param": {"type": "integer", "description": "This is the int parameter."},
                "literal_param": {
                    "type": "string",
                    "enum": ["option1", "option2"],
                    "description": "This is the literal parameter.",
                },
                "enum_param": {
                    "type": "string",
                    "enum": ["enum1", "enum2"],
                    "description": "This is the enum parameter.",
                },
                "default_str_param": {
                    "type": "string",
                    "description": "This is the default string parameter.",
                },
            },
            "required": ["str_param", "float_param", "int_param", "literal_param", "enum_param"],
        },
    }
    expected = CallableFunction(
        name="test_function",
        params_model=params_model,
        function=test_function,
        api_dict=api_dict,
        doc_str=doc_str,
    )
    exclude = {"doc_str", "params_model"}
    # -- Asserts --
    assert callable_function.dict(exclude=exclude) == expected.dict(
        exclude=exclude
    ), f"callable_function: {callable_function}\nexpected: {expected}"

    for field_key in callable_function.params_model.__fields__.keys():
        for property in ["name", "type_", "default", "required", "alias"]:
            assert getattr(callable_function.params_model.__fields__[field_key], property) == getattr(
                expected.params_model.__fields__[field_key], property
            ), f"callable_function: {callable_function}\nexpected: {expected}"
