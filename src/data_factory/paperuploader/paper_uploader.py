from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PointStruct

from data_factory.enrichjsons.enriched_paper import EnrichedPaper


@dataclass
class QdrantPointStructConverter:
    def from_enriched_paper(self, paper: EnrichedPaper) -> PointStruct:
        payload = paper.dumps()
        keys_to_drop = ["vectors", "fulltext"]
        for key_to_drop in keys_to_drop:
            del payload[key_to_drop]

        # Flatten author names
        payload["author_names"] = [
            name
            for author in payload["authors"]
            for name in [author["first"], author["last"]]
            if name is not None
        ]

        # Flatten named entities
        for key, entities in payload["named_entities"].items():
            payload[f"{key}_entities"] = entities
        del payload["named_entities"]

        # Flatten FoSs
        foss = [fos_item["label"] for fos_item in payload["field_of_studies"]]
        payload["field_of_studies"] = foss

        ps = PointStruct(
            id=paper.paper_uuid,
            vector={
                "overview": paper.vectors["overview"],
                "challenge": paper.vectors["challenge"]
                if paper.vectors["challenge"]
                else paper.vectors["overview"],
                "approach": paper.vectors["approach"]
                if paper.vectors["approach"]
                else paper.vectors["overview"],
                "outcome": paper.vectors["outcome"]
                if paper.vectors["outcome"]
                else paper.vectors["overview"],
            },
            payload=payload,
        )
        return ps


@dataclass
class PaperUploader:
    converter: QdrantPointStructConverter
    paper_collection_name: str
    vector_size: int
    client: QdrantClient

    def __post_init__(self):
        # make sure if the collection exists
        try:
            self.client.get_collection(self.paper_collection_name)
        except UnexpectedResponse:
            self.client.recreate_collection(
                collection_name=self.paper_collection_name,
                vectors_config={
                    "overview": models.VectorParams(
                        size=self.vector_size, distance=models.Distance.COSINE
                    ),
                    "challenge": models.VectorParams(
                        size=self.vector_size, distance=models.Distance.COSINE
                    ),
                    "approach": models.VectorParams(
                        size=self.vector_size, distance=models.Distance.COSINE
                    ),
                    "outcome": models.VectorParams(
                        size=self.vector_size, distance=models.Distance.COSINE
                    ),
                },
            )

            for field_name in [
                "material_entities",
                "method_entities",
                "metric_entities",
                "task_entities",
                "field_of_studies",
                "author_names",
            ]:
                self.client.create_payload_index(
                    collection_name=self.paper_collection_name,
                    field_name=field_name,
                    field_schema="keyword",
                )

    def upload(self, paper: EnrichedPaper) -> None:
        point = self.converter.from_enriched_paper(paper)
        self.client.upsert(collection_name=self.paper_collection_name, points=[point])

    def batch_upload(self, papers: list[EnrichedPaper]) -> None:
        points = [self.converter.from_enriched_paper(paper) for paper in papers]
        self.client.upsert(collection_name=self.paper_collection_name, points=points)
