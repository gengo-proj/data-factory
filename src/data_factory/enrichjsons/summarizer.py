from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Literal

from transformers import AutoTokenizer, pipeline

from data_factory.paper2json.raw_paper import RawPaper

SUMMARY_TYPES = ["overview", "challenge", "approach", "outcome"]


class BaseSummarizer(metaclass=ABCMeta):
    def __prepara_input(self, raw_paper: RawPaper) -> str | None:
        raise NotImplementedError("You cannot execute this method.")

    @abstractmethod
    def __call__(self, raw_paper: RawPaper) -> dict[str, str | None]:
        raise NotImplementedError("You cannot execute this method.")


class PLME2ESummarizer(BaseSummarizer):
    min_length: int = 8
    max_length: int = 256
    num_beams: int = 1
    device: int = -1

    def __init__(self):
        self.overview_summarizer = pipeline(
            "summarization",
            model="sobamchan/bart-large-scitldr",
            tokenizer=AutoTokenizer.from_pretrained(
                "sobamchan/bart-large-scitldr",
                model_max_length=512,
                max_length=512,
                truncation=True,
            ),
            device=self.device,
        )
        self.challenge_summarizer = pipeline(
            "summarization",
            model="sobamchan/t5-base-aclsum-challenge-nofilter",
            tokenizer=AutoTokenizer.from_pretrained(
                "t5-base",
                model_max_length=512,
                max_length=512,
                truncation=True,
            ),
            device=self.device,
        )
        self.approach_summarizer = pipeline(
            "summarization",
            model="sobamchan/t5-base-aclsum-approach-nofilter",
            tokenizer=AutoTokenizer.from_pretrained(
                "t5-base",
                model_max_length=512,
                max_length=512,
                truncation=True,
            ),
            device=self.device,
        )
        self.outcome_summarizer = pipeline(
            "summarization",
            model="sobamchan/t5-base-aclsum-outcome-nofilter",
            tokenizer=AutoTokenizer.from_pretrained(
                "t5-base",
                model_max_length=512,
                max_length=512,
                truncation=True,
            ),
            device=self.device,
        )

    def __prepara_input(self, raw_paper: RawPaper) -> str | None:
        texts: list[str] = []

        # abstract
        if raw_paper.abstract is not None:
            texts.append(raw_paper.abstract)

        if raw_paper.fulltext is not None:
            section_names = list(raw_paper.fulltext.keys())

            # introduction
            intro_sec_names = [
                section_name
                for section_name in section_names
                if section_name.find("introduction") != -1
            ]
            if intro_sec_names != []:
                intro_sents = raw_paper.fulltext[intro_sec_names[0]]
                texts.append(" ".join(intro_sents))

            # conclusion
            conclusion_sec_names = [
                section_name
                for section_name in section_names
                if section_name.find("conclusion") != -1
            ]
            if conclusion_sec_names != []:
                conclusion_sents = raw_paper.fulltext[conclusion_sec_names[0]]
                texts.append(" ".join(conclusion_sents))

            if len(texts) == 0:
                # when there are no introduction or conclusion, just use everything
                texts += [
                    sent
                    for sec_sents in raw_paper.fulltext.values()
                    for sent in sec_sents
                ]

        return " ".join(texts) if len(texts) != 0 else None

    def __summarize(
        self,
        text: str,
        summary_type: str,
    ) -> str:
        if summary_type == "overview":
            summary = self.overview_summarizer(
                text,
                truncation=True,
                min_length=self.min_length,
                max_length=self.max_length,
                num_beams=self.num_beams,
            )[0]["summary_text"]
        elif summary_type == "challenge":
            summary = self.challenge_summarizer(
                text,
                truncation=True,
                min_length=self.min_length,
                max_length=self.max_length,
                num_beams=self.num_beams,
            )[0]["summary_text"]
        elif summary_type == "approach":
            summary = self.approach_summarizer(
                text,
                truncation=True,
                min_length=self.min_length,
                max_length=self.max_length,
                num_beams=self.num_beams,
            )[0]["summary_text"]
        else:
            summary = self.outcome_summarizer(
                text,
                truncation=True,
                min_length=self.min_length,
                max_length=self.max_length,
                num_beams=self.num_beams,
            )[0]["summary_text"]
        return summary

    def __call__(self, raw_paper: RawPaper) -> dict[str, str | None]:
        aic = self.__prepara_input(raw_paper)
        abst = raw_paper.abstract

        summaries: dict[str, str | None] = {}
        for summary_type in SUMMARY_TYPES:
            input_text = abst if summary_type == "overview" else aic
            if input_text is None:
                summaries[summary_type] = None
            else:
                summaries[summary_type] = self.__summarize(input_text, summary_type)

        return summaries


@dataclass
class SummarizerFactory:
    model_type: Literal["plm-e2e"]
    device: int

    def load(self) -> BaseSummarizer:
        if self.model_type == "plm-e2e":
            return PLME2ESummarizer()
        else:
            raise ValueError(f"{self.model_type} is not supported.")
