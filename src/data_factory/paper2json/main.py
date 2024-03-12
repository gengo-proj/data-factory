import os
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path

import requests
import sienna
from acl_anthology import Anthology
from spacy.lang.en import English

from data_factory.paper2json.doc2json.process_pdf import process_pdf_file
from data_factory.paper2json.raw_paper import RawPaper
from data_factory.utils import paper_url_to_uuid

nlp = English()
nlp.add_pipe("sentencizer")


@dataclass
class PDFDownloader:
    def download(self, url: str, opath: str | Path) -> None:
        """Download a pdf file from URL and save locally.
        Skip if there is a file at `opath` already.

        Parameters
        ----------
        url : str
            URL of the target PDF file
        opath : str
            Path to save downloaded PDF data.
        """
        if os.path.exists(opath):
            return

        print(f"Downloading {url} into {opath}")
        with open(opath, "wb") as f:
            res = requests.get(url)
            f.write(res.content)


@dataclass
class SentenceTokenizer:
    nlp: English = field(init=False)

    def __post_init__(self):
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")

    def __call__(self, text: str) -> list[str]:
        return [sent.text for sent in nlp(text).sents]


@dataclass
class FulltextExtractor:
    sentence_tokenizer: SentenceTokenizer = field(default_factory=SentenceTokenizer)
    tmp_dir: str = "./tmp/grobid"

    def __call__(self, pdf_file_path: Path) -> tuple[dict[str, list[str]], str] | None:
        """Extract fulltext from a PDf file
        ```
        {
            "introduction": ["first sentence", "second sentence"],
            ...
            "conclusion": ["first sentence", "second sentence"],
        }
        ```

        Parameters
        ----------
        pdf_file_path : Path
            Source PDF file

        Returns
        -------
        tuple[dict[str, list[str]], str] | None
            `section_name` -> list[`sentence`], returns None if extraction fails, and abstract
        """
        try:
            extraction_fpath = process_pdf_file(
                str(pdf_file_path), temp_dir=self.tmp_dir, output_dir=self.tmp_dir
            )
            extraction_result = sienna.load(extraction_fpath)
            sections: dict[str, list[str]] = {}

            print("Segmenting texts into sentences by sections")
            for body_text in extraction_result["pdf_parse"]["body_text"]:
                section_name = body_text["section"].lower()

                sents = self.sentence_tokenizer(body_text["text"])

                if section_name not in sections.keys():
                    sections[section_name] = []

                sections[section_name] += sents

            return sections, extraction_result["abstract"]
        except AssertionError:
            print(f"Grobid failed to parse this document.")
            return None


@dataclass
class XML2Jsons:
    """Generate paper json files from a collection xml file, with fulltext extraction.

    - 1. extract relevant paper info from one item in xml file
    - 2. download pdf file
    - 3. extract fulltext
    - 4. segment sentences with spacy
    - 5. format a json file and save

    Parameters
    ----------
    xml_path : str
        Path to the original ACL xml file
    base_output_dir : str
        Dir to save all the paper json files
    """

    base_output_dir: Path
    pdf_output_dir: Path
    anthology: Anthology
    collection_id_filters: list[str] | None

    pdf_downloader: PDFDownloader = field(default_factory=PDFDownloader)
    fulltext_extractor: FulltextExtractor = field(default_factory=FulltextExtractor)

    @classmethod
    def from_cli(cls) -> "XML2Jsons":
        parser = ArgumentParser()
        parser.add_argument("--base-output-dir", type=str)
        parser.add_argument("--pdf-output-dir", type=str)
        parser.add_argument("--anthology-data-dir", type=str)
        parser.add_argument(
            "--collection-id-filters", nargs="+", type=str, default=None
        )
        args = parser.parse_args()
        return cls(
            base_output_dir=Path(args.base_output_dir),
            pdf_output_dir=Path(args.pdf_output_dir),
            anthology=Anthology(datadir=args.anthology_data_dir),
            collection_id_filters=args.collection_id_filters,
        )

    def run(self):
        for collection_id, collection in self.anthology.collections.items():
            if self.collection_id_filters is not None:
                if not any(
                    [
                        collection_id.find(filter_str) != -1
                        for filter_str in self.collection_id_filters
                    ]
                ):
                    continue
            print(f"Processing collection: {collection_id}")
            for volume in collection.volumes():
                volume_id = f"{collection_id}-{volume.id}"
                volume_dir = self.base_output_dir / collection_id / volume.id

                if not volume_dir.exists():
                    volume_dir.mkdir(parents=True, exist_ok=True)

                for paper in volume.papers():
                    if (paper.pdf is not None) and (paper.pdf.name.find("http") == -1):
                        pdf_save_path = self.pdf_output_dir / f"{paper.pdf.name}.pdf"
                        self.pdf_downloader.download(paper.pdf.url, pdf_save_path)
                        fulltext_extraction_output = self.fulltext_extractor(
                            pdf_save_path
                        )
                        url = paper.pdf.name
                        if fulltext_extraction_output:
                            fulltext, abstract = fulltext_extraction_output
                        else:
                            fulltext, abstract = None, None
                    else:
                        url = (
                            f"{volume_id}.{paper.id.rjust(3, '0')}"
                            if len(collection_id) == 1
                            else f"{volume_id}.{paper.id}"
                        )
                        fulltext, abstract = None, None

                    paper_uuid = paper_url_to_uuid(url)
                    raw_paper = RawPaper(
                        paper_uuid=str(paper_uuid),
                        collection_id=collection_id,
                        collection_acronym=volume.venues()[0].acronym,
                        volume_id=volume_id,
                        booktitle=volume.title.as_text(),
                        paper_id=int(paper.id),
                        year=int(paper.year),
                        paper_title=paper.title.as_text(),
                        authors=[
                            {"first": author.first, "last": author.last}
                            for author in paper.authors
                        ],
                        abstract=paper.abstract.as_text()
                        if paper.abstract is not None
                        else abstract,
                        url=url,
                        bibkey=paper.bibkey if paper.bibkey is not None else None,
                        doi=paper.doi if paper.doi is not None else None,
                        fulltext=fulltext,
                    )
                    raw_paper.save(str(volume_dir))


if __name__ == "__main__":
    xml2jsons = XML2Jsons.from_cli()
    xml2jsons.run()
