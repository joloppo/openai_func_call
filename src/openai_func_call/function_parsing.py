from inspect import signature
from typing import Callable, Optional
from warnings import warn

from docstring_parser import Docstring, parse
from pydantic import create_model


def create_model_for_func_params(func: Callable):
    sig = signature(func)
    parameters = sig.parameters

    fields = {}
    for param_name, param in parameters.items():
        if param.annotation != param.empty:
            # TODO: Could maybe add docstring as description to field
            if param.default != param.empty:
                fields[param_name] = (param.annotation, param.default)
            else:
                fields[param_name] = (param.annotation, ...)
        else:
            raise ValueError(f"Parameter '{param_name}' is missing a type annotation")

    return create_model("FunctionArgsModel", **fields)


def get_viable_docstring(func: Callable) -> Docstring:
    # check that there is a docstring
    if func.__doc__ is None:
        raise ValueError(f"Function {func.__name__} has no docstring.")

    # check that there is a short_description
    if not parse(func.__doc__).short_description:
        raise ValueError(f"Function {func.__name__} has no short description in its docstring.")

    # check that every parameter of function is in docstring
    for param_name in signature(func).parameters.keys():
        if param_name not in [param.arg_name for param in parse(func.__doc__).params]:
            # TODO: API might not need a description for every parameter
            raise ValueError(f"Docstring in function '{func.__name__}' missing parameter: {param_name}")

    # check that every parameter has a description
    for param in parse(func.__doc__).params:
        if not param.description:
            warn(f"Parameter {param.arg_name} of function {func.__name__} has no description.")

    return parse(func.__doc__)


if __name__ == "__main__":
    # Example function with arguments
    def example_func(name: str, age: int = 30, email: str = "", aoeu: Optional[str] = None):
        pass

    # Create Pydantic model from function arguments
    model = create_model_for_func_params(example_func)

    # Usage example
    data = {"name": "John", "age": 25, "email": "john@example.com"}

    instance = model(**data)
    print(instance.dict())
    print(instance)
    print(model.__fields__)
