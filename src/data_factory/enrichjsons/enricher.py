from dataclasses import dataclass

from data_factory.enrichjsons.enriched_paper import EnrichedPaper
from data_factory.enrichjsons.fos_classifier import FoSClassifier
from data_factory.enrichjsons.nerer import BaseNerer
from data_factory.enrichjsons.summarizer import BaseSummarizer
from data_factory.enrichjsons.vectorizer import DensePLMVectorizer
from data_factory.paper2json.raw_paper import RawPaper


@dataclass
class BasicEnricher:
    summarizer: BaseSummarizer
    nerer: BaseNerer
    vectorizer: DensePLMVectorizer
    fos_classifier: FoSClassifier

    def __call__(self, raw_paper: RawPaper) -> EnrichedPaper:
        summaries: dict[str, str | None] = self.summarizer(raw_paper)
        named_entities: dict[str, list[str]] = self.nerer(raw_paper)
        vectors: dict[str, list[float]] = self.vectorizer(raw_paper, summaries)
        field_of_studies: list[dict[str, str | float]] | None = self.fos_classifier(
            raw_paper
        )

        # Run custom module here if needed.

        _relevant_papers: dict[str, list[float] | None] = {
            "overview": None,
            "challenge": None,
            "approach": None,
            "outcome": None,
        }
        return EnrichedPaper(
            **raw_paper.dumps(),
            named_entities=named_entities,
            summaries=summaries,
            vectors=vectors,
            relevant_papers=_relevant_papers,
            field_of_studies=field_of_studies,
            # pass results of custom modules if needed
        )


@dataclass
class EnricherFactory:
    enricher_type: str
    summarier: BaseSummarizer
    nerer: BaseNerer
    vectorizer: DensePLMVectorizer
    fos_classifier: FoSClassifier

    def load(self) -> BasicEnricher:
        if self.enricher_type == "basic":
            print("Initializing a basic enricher")
            return BasicEnricher(
                summarizer=self.summarier,
                nerer=self.nerer,
                vectorizer=self.vectorizer,
                fos_classifier=self.fos_classifier,
            )
        else:
            raise ValueError(f"{self.enricher_type} is not supported.")
