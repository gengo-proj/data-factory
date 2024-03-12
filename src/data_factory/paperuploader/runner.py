from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from data_factory.enrichjsons.enriched_paper import EnrichedPaper
from data_factory.paperuploader.paper_uploader import (
    PaperUploader, QdrantPointStructConverter)


@dataclass
class Runner:
    paper_base_dir: Path
    paper_uploader: PaperUploader
    batch_size: int

    @classmethod
    def from_cli(cls):
        parser = ArgumentParser()
        parser.add_argument("--paper-base-dir", type=str, required=True)
        parser.add_argument("--collection-name", type=str, required=True)
        parser.add_argument("--vector-size", type=int, required=True)
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--host", type=str, default="localhost")
        parser.add_argument("--port", type=int, default=6333)
        parser.add_argument("--api-key", type=str, default=None, required=False)
        args = parser.parse_args()

        QDRANT_HOST = args.host
        QDRANT_PORT = args.port
        QDRANT_COLLECTION_NAME = args.collection_name

        return cls(
            paper_base_dir=Path(args.paper_base_dir),
            paper_uploader=PaperUploader(
                converter=QdrantPointStructConverter(),
                paper_collection_name=QDRANT_COLLECTION_NAME,
                vector_size=args.vector_size,
                client=QdrantClient(
                    QDRANT_HOST, port=QDRANT_PORT, timeout=100, api_key=args.api_key
                ),
            ),
            batch_size=args.batch_size,
        )

    def collect_paper_paths(self) -> list[Path]:
        return list(self.paper_base_dir.glob("**/*.json"))

    def run(self):
        paper_paths = self.collect_paper_paths()
        for idx in range(0, len(paper_paths), self.batch_size):
            batch_paper_paths = paper_paths[idx : idx + self.batch_size]
            batch_papers = [
                EnrichedPaper.load_from_json(paper_path)
                for paper_path in batch_paper_paths
            ]
            self.paper_uploader.batch_upload(batch_papers)


if __name__ == "__main__":
    runner = Runner.from_cli()
    runner.run()
