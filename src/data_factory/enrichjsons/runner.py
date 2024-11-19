from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from data_factory.enrichjsons.enriched_paper import EnrichedPaper
from data_factory.enrichjsons.enricher import BasicEnricher, EnricherFactory, EnrichType
from data_factory.enrichjsons.fos_classifier import FoSClassifierFactory
from data_factory.enrichjsons.nerer import NererFactory
from data_factory.enrichjsons.summarizer import SummarizerFactory
from data_factory.enrichjsons.vectorizer import VectorizerFactory
from data_factory.paper2json.raw_paper import RawPaper
from data_factory.utils import read_toml


@dataclass
class Runner:
    raw_paper_base_dir: Path
    output_base_dir: Path
    enricher: BasicEnricher

    @classmethod
    def from_cli(cls) -> "Runner":
        parser = ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument("--raw-paper-base-dir", type=str, required=True)
        parser.add_argument("--output-base-dir", type=str, required=True)
        parser.add_argument(
            "--update-enrich-types",
            nargs="+",
            type=EnrichType,
            default=[],
            required=False,
            help=f"Select from [{', '.join(x for x in EnrichType)}]",
        )
        parser.add_argument("--device", type=int, default=-1)
        args = parser.parse_args()

        params = read_toml(args.config)
        summarization_params = params["summarization"]
        ner_params = params["ner"]
        vectorizer_params = params["vectorizer"]

        return cls(
            raw_paper_base_dir=Path(args.raw_paper_base_dir),
            output_base_dir=Path(args.output_base_dir),
            enricher=EnricherFactory(
                enricher_type=params["enricher_type"],
                summarier=SummarizerFactory(
                    summarization_params["model_type"], device=args.device
                ).load(),
                nerer=NererFactory(ner_params["model_type"], device=args.device).load(),
                vectorizer=VectorizerFactory(
                    model_type=vectorizer_params["model_type"], device=args.device
                ).load(),
                fos_classifier=FoSClassifierFactory(device=args.device).load(),
                update_enrich_types=args.update_enrich_types,
            ).load(),
        )

    def collect_raw_paper_paths(self) -> list[Path]:
        return list(self.raw_paper_base_dir.glob("**/*.json"))

    def raw_paper_path_to_new_dir(self, raw_paper_path: Path) -> Path:
        collection_name = raw_paper_path.parent.parent.name
        volume_name = raw_paper_path.parent.name
        new_path = Path(self.output_base_dir, collection_name, volume_name)
        if not new_path.exists():
            new_path.mkdir(parents=True)
        assert new_path.is_dir(), "This must be a dir to save a paper json file."
        return new_path

    def run(self) -> None:
        for raw_paper_fpath in tqdm(self.collect_raw_paper_paths()):
            raw_paper = RawPaper.load_from_json(raw_paper_fpath)
            odir = self.raw_paper_path_to_new_dir(raw_paper_fpath)

            existing_enriched_paper = (
                EnrichedPaper.load_from_json(odir / raw_paper.get_fname())
                if (odir / raw_paper.get_fname()).exists()
                else None
            )
            enriched_paper = self.enricher(raw_paper, existing_enriched_paper)
            enriched_paper.save(odir)


if __name__ == "__main__":
    runner = Runner.from_cli()
    runner.run()
