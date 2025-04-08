from argparse import ArgumentParser
from dataclasses import dataclass, field
from iclr_downloader import get_proceeding as get_iclr_proceeding
from neurips_downloader.core import get_publication_list as get_neurips_publication_list
from enum import StrEnum
from pathlib import Path

from data_factory.paper2json.main import FulltextExtractor, PDFDownloader
from data_factory.paper2json.raw_paper import RawPaper
from data_factory.utils import paper_url_to_uuid


class VenueType(StrEnum):
    iclr = "iclr"
    neurips = "neurips"


@dataclass
class Author:
    first: str
    last: str | None

    @classmethod
    def from_string(cls, name_str: str) -> "Author":
        name_parts = name_str.split()
        if len(name_parts) == 1:
            first = name_parts[0]
            last = None
        else:
            first = " ".join(name_parts[:-1])
            last = name_parts[-1]
        return cls(first, last)


@dataclass(frozen=True)
class Venue:
    _type: VenueType
    year: int

    @property
    def collection_id(self) -> str:
        return f"{self.year}.{self._type}"

    @property
    def collection_acronym(self) -> str:
        match self._type:
            case VenueType.neurips:
                return "NIPS" if self.year < 2018 else "NeurIPS"
            case VenueType.iclr:
                return "ICLR"

    @property
    def volume_id(self) -> str:
        # Change the last part (`conference`) when adding more types, e.g., workshop
        return f"{self.year}.{self._type}-conference"

    @property
    def booktilte(self) -> str:
        match self._type:
            case VenueType.neurips:
                idx = (self.year - 1988) + 1
                if self.year < 2018:
                    return f"Advances in Neural Information Processing Systems {idx} (NIPS {self.year})"
                else:
                    return f"Advances in Neural Information Processing Systems {idx} (NeurIPS {self.year})"
            case VenueType.iclr:
                return f"International Conference on Learning Representations (ICLR {self.year})"


@dataclass
class Conf2Jsons:
    base_output_dir: Path
    pdf_output_dir: Path
    venue: Venue
    username: str | None
    password: str | None

    pdf_downloader: PDFDownloader = field(default_factory=PDFDownloader)
    fulltext_extractor: FulltextExtractor = field(default_factory=FulltextExtractor)

    @classmethod
    def from_cli(cls) -> "Conf2Jsons":
        parser = ArgumentParser()
        parser.add_argument("--base-output-dir", type=Path)
        parser.add_argument("--pdf-output-dir", type=Path)
        parser.add_argument("--venue-type", type=VenueType)
        parser.add_argument("--year", type=int)
        parser.add_argument("--username", type=str, default=None)
        parser.add_argument("--password", type=str, default=None)
        args = parser.parse_args()

        if args.venue_type == VenueType.iclr:
            if (args.username is None) or (args.password is None):
                raise ValueError("For ICLR, you need to pass `username` and `password`")

        venue = Venue(args.venue_type, args.year)

        return cls(
            base_output_dir=args.base_output_dir,
            pdf_output_dir=args.pdf_output_dir,
            venue=venue,
            username=args.username,
            password=args.password,
        )

    def run(self) -> None:
        match self.venue._type:
            case VenueType.iclr:
                assert isinstance(self.username, str)
                assert isinstance(self.password, str)
                papers = get_iclr_proceeding(
                    self.venue.year, "Conference", self.username, self.password
                )
            case VenueType.neurips:
                papers = get_neurips_publication_list(self.venue.year)

        volume_dir = (
            self.base_output_dir / self.venue.collection_id / self.venue.volume_id
        )
        if not volume_dir.exists():
            volume_dir.mkdir(parents=True, exist_ok=True)

        for paper in papers:
            pdf_save_path = (
                self.pdf_output_dir / f"{self.venue.volume_id}.{paper.id}.pdf"
            )
            self.pdf_downloader.download(paper.pdf_url, pdf_save_path)
            fulltext_extraction_output = self.fulltext_extractor(pdf_save_path)
            if fulltext_extraction_output:
                fulltext, _ = fulltext_extraction_output
            else:
                fulltext = None
            abstract = paper.abstract

            authors = [Author.from_string(author_str) for author_str in paper.authors]

            url = f"{self.venue.volume_id}.{paper.id}"
            raw_paper = RawPaper(
                paper_uuid=str(paper_url_to_uuid(url)),
                collection_id=self.venue.collection_id,
                collection_acronym=self.venue.collection_acronym,
                volume_id=self.venue.volume_id,
                booktitle=self.venue.booktilte,
                paper_id=paper.id,
                year=int(paper.year),
                paper_title=paper.title,
                authors=[
                    {"first": author.first, "last": author.last} for author in authors
                ],
                abstract=abstract,
                url=url,
                bibkey=None,
                doi=None,
                fulltext=fulltext,
            )
            raw_paper.save(str(volume_dir))


if __name__ == "__main__":
    c2j = Conf2Jsons.from_cli()
    c2j.run()
