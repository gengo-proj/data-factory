import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sienna
from jsonschema import validate

from data_factory.enrichjsons.custom_module import CustomResultValue

schema = sienna.load("./src/data_factory/jsonschemas/paper_enriched_schema.json")
assert isinstance(schema, dict)


@dataclass
class EnrichedPaper:
    paper_uuid: str

    collection_id: str
    collection_acronym: str
    volume_id: str
    booktitle: str
    paper_id: int
    year: int

    paper_title: str
    authors: list[dict[str, str | None]]
    abstract: str
    url: str
    bibkey: str
    doi: str
    fulltext: dict[str, list[str]] | None

    named_entities: dict[str, list[str]]
    summaries: dict[str, str | None]
    field_of_studies: list[dict[str, str | float]] | None

    vectors: dict[str, list[float]]
    relevant_papers: dict[str, list[float] | None]

    custom_results: list[CustomResultValue] | None = None

    @classmethod
    def load_from_json(cls, fpath: str | Path) -> "EnrichedPaper":
        fpath = fpath if not isinstance(fpath, Path) else str(fpath)
        return cls(**sienna.load(fpath))

    def get_name(self) -> str:
        return str(self.url)

    def dumps(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        validate(self.dumps(), schema=schema)

    def save(self, odir: str | Path) -> None:
        self.validate()
        opath = os.path.join(odir, f"{self.get_name()}.json")
        sienna.save(self.dumps(), opath, indent=2)
