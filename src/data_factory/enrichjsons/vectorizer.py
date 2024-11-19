from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from sentence_transformers import SentenceTransformer

from data_factory.paper2json.raw_paper import RawPaper


class DensePLMVectorizer:
    def __init__(self, device: int):
        self.model = SentenceTransformer(
            "Snowflake/snowflake-arctic-embed-m",
            device="cuda" if device > -1 else "cpu",
        )
        self.device = device

    def encode(self, text: str) -> list[float]:
        return cast(np.ndarray, self.model.encode(text)).tolist()

    def __call__(
        self, raw_paper: RawPaper, summaries: dict[str, str | None]
    ) -> dict[str, list[float]]:
        # Use vector generated for overview for other aspects when summaries are not available
        overview_vec = (
            self.encode(raw_paper.paper_title + " " + raw_paper.abstract)
            if raw_paper.abstract
            else self.encode(raw_paper.paper_title)
        )
        return {
            "overview": overview_vec,
            "challenge": self.encode(summaries["challenge"])
            if summaries["challenge"]
            else overview_vec,
            "approach": self.encode(summaries["approach"])
            if summaries["approach"]
            else overview_vec,
            "outcome": self.encode(summaries["outcome"])
            if summaries["outcome"]
            else overview_vec,
        }


@dataclass
class VectorizerFactory:
    model_type: Literal["dense-plm"]
    device: int

    def load(self) -> DensePLMVectorizer:
        if self.model_type == "dense-plm":
            return DensePLMVectorizer(device=self.device)
        else:
            raise ValueError(f"{self.model_type} is not supported.")
