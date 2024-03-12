from abc import ABCMeta, abstractmethod
from typing import TypedDict

from data_factory.paper2json.raw_paper import RawPaper


class CustomResultValue(TypedDict):
    result_name: str
    result_value: str


class BaseCustomModule(metaclass=ABCMeta):
    def __init__(self, result_name: str):
        self.result_name = result_name

    @abstractmethod
    def __call__(self, raw_paper: RawPaper) -> CustomResultValue:
        raise NotImplementedError("This method is not yet implemented.")


class ExampleCustomModule(BaseCustomModule):
    """An exmple of how to implement custom enrichment module.
    This example simply returns the paper's title."""

    def __init__(self):
        super().__init__(result_name="paper_title")

    def __call__(self, raw_paper: RawPaper) -> CustomResultValue:
        return {"result_name": self.result_name, "result_value": raw_paper.paper_title}
