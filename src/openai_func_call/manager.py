import json
import logging
from typing import Callable, Optional

import openai
from pydantic import BaseModel

from openai_func_call.assembly import CallableFunction, func_to_callable_function
from openai_func_call.schemas import FinishReason

logger = logging.getLogger(__name__)


class APIError(Exception):
    pass


class FunctionManagerError(Exception):
    pass


class QueryState(BaseModel):
    total_tokens_used: int = 0
    total_functions_called: int = 0
    max_func_calls: Optional[int] = 1000
    max_tokens_used: Optional[int] = None

    class Config:
        validate_assignment = True

    @property
    def should_stop(self) -> bool:
        if self.max_tokens_used is not None and self.total_tokens_used > self.max_tokens_used:
            return True
        if self.max_func_calls is not None and self.total_functions_called > self.max_func_calls:
            return True
        return False


class FunctionManager:
    max_func_calls: Optional[int]
    max_token_calls: Optional[int]

    def __init__(
        self, functions: list[Callable], max_func_calls: Optional[int] = None, max_tokens_used: Optional[int] = None
    ):
        """Create a function manager with the given functions."""
        # check
        func_names = [func.__name__ for func in functions]
        assert len(func_names) == len(set(func_names)), "Function names must be unique."

        callable_functions = [func_to_callable_function(func) for func in functions]
        self.callable_functions_dict: dict[str, CallableFunction] = {func.name: func for func in callable_functions}
        self.max_func_calls = max_func_calls
        self.max_tokens_used = max_tokens_used

    def call_function(self, func_name: str, **kwargs):
        """Call a function by name with the given kwargs.

        Kwargs will be checked using a pydantic model created for the parameters, then call the function.
        """
        parsed_args_model = self.callable_functions_dict[func_name].params_model(**kwargs)
        func = self.callable_functions_dict[func_name].function
        return func(**parsed_args_model.dict())

    @property
    def api_function_dict(self) -> list[dict]:
        return [callable_function.api_dict for name, callable_function in self.callable_functions_dict.items()]

    def query_openai(self, query: str, model: str = "gpt-3.5-turbo-0613", function_call: str = "auto", **kwargs):
        """Query the OpenAI API with the given kwargs."""
        messages = [{"role": "user", "content": query}]
        response = None
        state = QueryState(max_func_calls=self.max_func_calls, max_tokens_used=self.max_tokens_used)
        # stop conditions
        args = dict(
            model=model,
            messages=messages,
            functions=self.api_function_dict,
            function_call=function_call,
        )
        args.update(kwargs)
        while not state.should_stop:
            response = openai.ChatCompletion.create(**args)

            tokens_used = response.get("usage", {}).get("total_tokens", None)
            if tokens_used is None:
                raise APIError("OpenAI response did not contain tokens used.")
            state.total_tokens_used += tokens_used

            # check
            if "choices" not in response or len(response["choices"]) == 0:  # TODO: what if choices > 1?
                raise APIError("OpenAI response did not contain any choices.")
            if "finish_reason" not in response["choices"][0]:
                raise APIError("OpenAI response did not contain a finish reason.")

            finish_reason = FinishReason(response["choices"][0]["finish_reason"])

            # errors
            if finish_reason == FinishReason.LENGTH:
                logger.info("OpenAI response finished due to length.")
                logger.info(response)
                raise APIError("API errored with length finish reason.")
            elif finish_reason == FinishReason.CONTENT_FILTER:
                raise APIError("Content filter finish reason not implemented.")

            # check
            if "message" not in response["choices"][0]:
                raise APIError("OpenAI response did not contain a message.")
            if "content" not in response["choices"][0]["message"]:
                raise APIError("OpenAI response did not contain a message content.")

            message = response["choices"][0]["message"]
            message_content = message["content"]

            if finish_reason == FinishReason.STOP:
                if not message_content:
                    raise APIError("OpenAI response finished with stop reason, but message had no content.")
                return message_content

            elif finish_reason == FinishReason.FUNCTION_CALL:
                state.total_functions_called += 1
                function_name = message.get("function_call", {}).get("name", "")
                if not function_name:
                    raise APIError("OpenAI response finished with function call, but message had no function call.")
                arg_str = message.get("function_call", {}).get("arguments", "")
                function_args = json.loads(arg_str)
                function_response = self.call_function(function_name, **function_args)
                messages.append({"role": "function", "name": function_name, "content": function_response})
            else:
                raise NotImplementedError(f"Finish reason {finish_reason} not implemented.")

        raise FunctionManagerError(f"Function manager stopped due to state: {state.dict()}")

    @staticmethod
    def continue_openai_query():  # TODO: save incase something breaks
        ...
