from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sienna

from data_factory.enrichjsons.enriched_paper import EnrichedPaper


@dataclass(frozen=True)
class Collection:
    collection_id: str
    collection_name: str
    collection_long_name: str
    year: int
    is_acl: bool
    is_toplevel: bool
    acronym: str
    old_styleletter: str | None
    volume_ids: list[str]
    event_id: str
    colocated_volume_ids: list[str]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "collection_long_name": self.collection_long_name,
            "year": self.year,
            "is_acl": self.is_acl,
            "is_toplevel": self.is_toplevel,
            "acronym": self.acronym,
            "old_styleletter": self.old_styleletter,
            "volume_ids": self.volume_ids,
            "event_id": self.event_id,
            "colocated_volume_ids": self.colocated_volume_ids,
        }


@dataclass(frozen=True)
class Volume:
    volume_id: str
    booktitle: str
    editors: list[dict[str, str]]
    address: str | None
    month: int | None
    year: int
    url: str
    venue: str
    frontmatter: str | None
    paper_ids: list[str]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "volume_id": self.volume_id,
            "booktitle": self.booktitle,
            "editors": self.editors,
            "address": self.address,
            "month": self.month,
            "year": self.year,
            "url": self.url,
            "venue": self.venue,
            "frontmatter": self.frontmatter,
            "paper_ids": self.paper_ids,
        }


Acronym2Longname = {
    "iclr": "International Conference on Learning Representations",
    "neurips": "Conference on Neural Information Processing Systems",
}


@dataclass(frozen=True)
class Conf2Metadata:
    enriched_paper_dir: Path
    output_dir: Path

    @classmethod
    def from_cli(cls) -> "Conf2Metadata":
        parser = ArgumentParser()
        parser.add_argument("--enriched-paper-dir", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        args = parser.parse_args()
        return cls(
            enriched_paper_dir=args.enriched_paper_dir,
            output_dir=args.output_dir,
        )

    def run(self) -> None:
        for collection_dpath in self.enriched_paper_dir.iterdir():
            collection_volume_ids: list[str] = []
            for volume_dpath in collection_dpath.iterdir():
                volume_paper_uuids = []
                one_paper = None
                for idx, paper_fpath in enumerate(volume_dpath.iterdir()):
                    paper = EnrichedPaper.load_from_json(paper_fpath)
                    volume_paper_uuids.append(paper.paper_uuid)
                    if idx == 0:
                        one_paper = paper

                assert one_paper
                volume = Volume(
                    volume_id=one_paper.volume_id,
                    booktitle=one_paper.booktitle,
                    editors=[],
                    address=None,
                    month=None,
                    year=one_paper.year,
                    url=one_paper.volume_id,
                    venue=one_paper.collection_acronym.lower(),
                    frontmatter=None,
                    paper_ids=volume_paper_uuids,
                )
                sienna.save(
                    volume.to_serializable(),
                    self.output_dir / "volumes" / f"{volume.volume_id}.json",
                )

                collection_volume_ids.append(volume.volume_id)

            # Make Collection
            assert one_paper
            collection = Collection(
                collection_id=one_paper.collection_id,
                collection_name=one_paper.collection_acronym.lower(),
                collection_long_name=Acronym2Longname[
                    one_paper.collection_acronym.lower()
                ],
                year=one_paper.year,
                is_acl=False,
                is_toplevel=True,
                acronym=one_paper.collection_acronym,
                old_styleletter=None,
                volume_ids=collection_volume_ids,  # When adding more volumes than conference, modify here.
                event_id=one_paper.collection_id,
                colocated_volume_ids=[],
            )
            sienna.save(
                collection.to_serializable(),
                self.output_dir / "collections" / f"{collection.collection_id}.json",
            )


if __name__ == "__main__":
    conf2meta = Conf2Metadata.from_cli()
    conf2meta.run()
