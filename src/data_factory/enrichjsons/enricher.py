from dataclasses import dataclass
from enum import StrEnum

from data_factory.enrichjsons.enriched_paper import EnrichedPaper
from data_factory.enrichjsons.fos_classifier import FoSClassifier
from data_factory.enrichjsons.nerer import BaseNerer
from data_factory.enrichjsons.summarizer import BaseSummarizer
from data_factory.enrichjsons.vectorizer import DensePLMVectorizer
from data_factory.paper2json.raw_paper import RawPaper


class EnrichType(StrEnum):
    summarization = "summarization"
    ner = "ner"
    vector = "vector"
    fos = "fos"


@dataclass
class BasicEnricher:
    summarizer: BaseSummarizer
    nerer: BaseNerer
    vectorizer: DensePLMVectorizer
    fos_classifier: FoSClassifier

    update_enrich_types: list[EnrichType]

    def __call__(
        self, raw_paper: RawPaper, existing_enriched_paper: EnrichedPaper | None
    ) -> EnrichedPaper:
        if (
            EnrichType.summarization not in self.update_enrich_types
        ) and existing_enriched_paper:
            summaries: dict[str, str | None] = existing_enriched_paper.summaries
        else:
            summaries = self.summarizer(raw_paper)

        if (EnrichType.ner not in self.update_enrich_types) and existing_enriched_paper:
            named_entities: dict[str, list[str]] = (
                existing_enriched_paper.named_entities
            )
        else:
            named_entities = self.nerer(raw_paper)

        if (
            EnrichType.vector not in self.update_enrich_types
        ) and existing_enriched_paper:
            vectors: dict[str, list[float]] = existing_enriched_paper.vectors
        else:
            vectors = self.vectorizer(raw_paper, summaries)

        if (EnrichType.fos not in self.update_enrich_types) and existing_enriched_paper:
            field_of_studies: list[dict[str, str | float]] | None = (
                existing_enriched_paper.field_of_studies
            )
        else:
            field_of_studies = self.fos_classifier(raw_paper)

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
    update_enrich_types: list[EnrichType]

    def load(self) -> BasicEnricher:
        if self.enricher_type == "basic":
            print("Initializing a basic enricher")
            return BasicEnricher(
                summarizer=self.summarier,
                nerer=self.nerer,
                vectorizer=self.vectorizer,
                fos_classifier=self.fos_classifier,
                update_enrich_types=self.update_enrich_types,
            )
        else:
            raise ValueError(f"{self.enricher_type} is not supported.")
