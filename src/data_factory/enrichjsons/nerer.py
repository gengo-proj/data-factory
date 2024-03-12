from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import spacy
from transformers import AutoTokenizer, pipeline

from data_factory.paper2json.raw_paper import RawPaper

NER_TYPES = ["material", "method", "metric", "task"]


class BaseNerer(metaclass=ABCMeta):
    def __prepara_input(self, raw_paper: RawPaper) -> str:
        raise NotImplementedError("You cannot execute this method.")

    @abstractmethod
    def __call__(self, raw_paper: RawPaper) -> dict[str, list[str]]:
        raise NotImplementedError("You cannot execute this method.")


class SpacyNerer(BaseNerer):
    def __init__(self):
        self.model = spacy.load(Path(__file__).parent / "models" / "spacy_ner")

    def __prepara_input(self, raw_paper: RawPaper) -> str:
        return raw_paper.abstract if raw_paper.abstract else raw_paper.paper_title

    def __call__(self, raw_paper: RawPaper) -> dict[str, list[str]]:
        text = self.__prepara_input(raw_paper)
        doc = self.model(text)
        results: dict[str, list[str]] = {ner_type: [] for ner_type in NER_TYPES}
        for ent in doc.ents:
            if ent.text not in results[ent.label_.lower()]:
                results[ent.label_.lower()].append(ent.text)
        return results


class PLMNerer(BaseNerer):
    def __init__(self, device: int):
        self.model = pipeline(
            "ner",
            model="anlp/SciERC_SciREX_trained_Scibert",
            tokenizer=AutoTokenizer.from_pretrained(
                "allenai/scibert_scivocab_uncased", model_max_length=512
            ),
            aggregation_strategy="average",
            device=device,
        )

    def __prepara_input(self, raw_paper: RawPaper) -> str:
        return raw_paper.abstract if raw_paper.abstract else raw_paper.paper_title

    def __call__(self, raw_paper: RawPaper) -> dict[str, list[str]]:
        inp = self.__prepara_input(raw_paper)
        ners = cast(list, self.model(inp))
        results: dict[str, list[str]] = {ner_type: [] for ner_type in NER_TYPES}
        for ner in ners:
            if ner["word"] not in results[ner["entity_group"].lower()]:
                results[ner["entity_group"].lower()].append(ner["word"])
        return results


@dataclass
class NererFactory:
    model_type: Literal["spacy", "plm"]
    device: int

    def load(self) -> BaseNerer:
        if self.model_type == "plm":
            return PLMNerer(device=self.device)
        elif self.model_type == "spacy":
            return SpacyNerer()
        else:
            raise ValueError(f"{self.model_type} is not supported.")
