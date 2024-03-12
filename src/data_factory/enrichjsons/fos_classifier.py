from dataclasses import dataclass
from typing import cast

from transformers import AutoTokenizer, pipeline

from data_factory.paper2json.raw_paper import RawPaper


class FoSClassifier:
    def __init__(self, device: int):
        self.model = pipeline(
            "text-classification",
            model="TimSchopf/nlp_taxonomy_classifier",
            tokenizer=AutoTokenizer.from_pretrained(
                "TimSchopf/nlp_taxonomy_classifier",
                model_max_length=512,
                max_length=512,
            ),
            device=device,
        )

    def __prepare_input(self, raw_paper: RawPaper) -> str:
        return (
            raw_paper.paper_title
            + self.model.tokenizer.sep_token
            + (raw_paper.abstract if raw_paper.abstract is not None else "")
        )

    def __call__(self, raw_paper: RawPaper) -> list[dict[str, str | float]] | None:
        inp = self.__prepare_input(raw_paper)
        preds = cast(list, self.model(inp, **{"truncation": True}))
        preds = [pred for pred in preds if pred["score"] >= 0.95]
        preds = sorted(preds, key=lambda pred: pred["score"], reverse=True)
        return preds


@dataclass
class FoSClassifierFactory:
    device: int

    def load(self) -> FoSClassifier:
        return FoSClassifier(self.device)
