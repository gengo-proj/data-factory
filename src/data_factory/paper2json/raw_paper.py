import sienna
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sienna
from jsonschema import validate

schema = sienna.load("./src/data_factory/jsonschemas/paper_raw_schema.json")
assert isinstance(schema, dict)


@dataclass
class RawPaper:
    paper_uuid: str

    collection_id: str
    collection_acronym: str
    volume_id: str
    booktitle: str
    paper_id: int | str
    year: int | None

    paper_title: str
    authors: list[dict[str, str | None]]
    abstract: str | None
    url: str
    bibkey: str | None
    doi: str | None
    fulltext: dict[str, list[str]] | None

    @classmethod
    def load_from_json(cls, fpath: str | Path) -> "RawPaper":
        fpath = fpath if not isinstance(fpath, Path) else str(fpath)
        return cls(**sienna.load(fpath))

    def get_name(self) -> str:
        if ("iclr" in self.collection_id) or ("neurips" in self.collection_id):
            return str(self.paper_uuid)
        else:
            return str(self.url)

    def get_fname(self) -> str:
        return f"{self.get_name()}.json"

    def dumps(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        validate(self.dumps(), schema=schema)

    def save(self, odir: str) -> None:
        self.validate()
        opath = os.path.join(odir, self.get_fname())
        sienna.save(self.dumps(), opath, indent=2)
