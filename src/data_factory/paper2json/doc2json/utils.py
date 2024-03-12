import glob
import io
import ntpath
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import bs4
from bs4 import BeautifulSoup, NavigableString

from data_factory.paper2json.doc2json.client import ApiClient
from data_factory.paper2json.doc2json.grobid_util import *

S2ORC_NAME_STRING = "S2ORC"
S2ORC_VERSION_STRING = "1.0.0"


CORRECT_KEYS = {"issn": "issue", "type": "type_str"}

SKIP_KEYS = {"link", "bib_id"}

REFERENCE_OUTPUT_KEYS = {
    "figure": {"text", "type_str", "uris", "num", "fig_num"},
    "table": {"text", "type_str", "content", "num", "html"},
    "footnote": {"text", "type_str", "num"},
    "section": {"text", "type_str", "num", "parent"},
    "equation": {"text", "type_str", "latex", "mathml", "num"},
}

METADATA_KEYS = {"title", "authors", "year", "venue", "identifiers"}


def replace_refspans(
    spans_to_replace: List[Tuple[int, int, str, str]],
    full_string: str,
    pre_padding: str = "",
    post_padding: str = "",
    btwn_padding: str = ", ",
) -> str:
    """
    For each span within the full string, replace that span with new text
    :param spans_to_replace: list of tuples of form (start_ind, end_ind, span_text, new_substring)
    :param full_string:
    :param pre_padding:
    :param post_padding:
    :param btwn_padding:
    :return:
    """
    # assert all spans are equal to full_text span
    assert all(
        [full_string[start:end] == span for start, end, span, _ in spans_to_replace]
    )

    # assert none of the spans start with the same start ind
    start_inds = [rep[0] for rep in spans_to_replace]
    assert len(set(start_inds)) == len(start_inds)

    # sort by start index
    spans_to_replace.sort(key=lambda x: x[0])

    # form strings for each span group
    for i, entry in enumerate(spans_to_replace):
        start, end, span, new_string = entry

        # skip empties
        if end <= 0:
            continue

        # compute shift amount
        shift_amount = (
            len(new_string) - len(span) + len(pre_padding) + len(post_padding)
        )

        # shift remaining appropriately
        for ind in range(i + 1, len(spans_to_replace)):
            next_start, next_end, next_span, next_string = spans_to_replace[ind]
            # skip empties
            if next_end <= 0:
                continue
            # if overlap between ref span and current ref span, remove from replacement
            if next_start < end:
                next_start = 0
                next_end = 0
                next_string = ""
            # if ref span abuts previous reference span
            elif next_start == end:
                next_start += shift_amount
                next_end += shift_amount
                next_string = btwn_padding + pre_padding + next_string + post_padding
            # if ref span starts after, shift starts and ends
            elif next_start > end:
                next_start += shift_amount
                next_end += shift_amount
                next_string = pre_padding + next_string + post_padding
            # save adjusted span
            spans_to_replace[ind] = (next_start, next_end, next_span, next_string)

    spans_to_replace = [entry for entry in spans_to_replace if entry[1] > 0]
    spans_to_replace.sort(key=lambda x: x[0])

    # apply shifts in series
    for start, end, span, new_string in spans_to_replace:
        assert full_string[start:end] == span
        full_string = full_string[:start] + new_string + full_string[end:]

    return full_string


class ReferenceEntry:
    """
    Class for representing S2ORC figure and table references

    An example json representation (values are examples, not accurate):

    {
      "FIGREF0": {
        "text": "FIG. 2. Depth profiles of...",
        "latex": null,
        "type": "figure"
      },
      "TABREF2": {
        "text": "Diversity indices of...",
        "latex": null,
        "type": "table",
        "content": "",
        "html": ""
      }
    }
    """

    def __init__(
        self,
        ref_id: str,
        text: str,
        type_str: str,
        latex: Optional[str] = None,
        mathml: Optional[str] = None,
        content: Optional[str] = None,
        html: Optional[str] = None,
        uris: Optional[List[str]] = None,
        num: Optional[str] = None,
        parent: Optional[str] = None,
        fig_num: Optional[str] = None,
    ):
        self.ref_id = ref_id
        self.text = text
        self.type_str = type_str
        self.latex = latex
        self.mathml = mathml
        self.content = content
        self.html = html
        self.uris = uris
        self.num = num
        self.parent = parent
        self.fig_num = fig_num

    def as_json(self):
        keep_keys = REFERENCE_OUTPUT_KEYS.get(self.type_str, None)
        if keep_keys:
            return {k: self.__getattribute__(k) for k in keep_keys}
        else:
            return {
                "text": self.text,
                "type": self.type_str,
                "latex": self.latex,
                "mathml": self.mathml,
                "content": self.content,
                "html": self.html,
                "uris": self.uris,
                "num": self.num,
                "parent": self.parent,
                "fig_num": self.fig_num,
            }


class BibliographyEntry:
    """
    Class for representing S2ORC parsed bibliography entries

    An example json representation (values are examples, not accurate):

    {
        "title": "Mobility Reports...",
        "authors": [
            {
                "first": "A",
                "middle": ["A"],
                "last": "Haija",
                "suffix": ""
            }
        ],
        "year": 2015,
        "venue": "IEEE Wireless Commun. Mag",
        "volume": "42",
        "issn": "9",
        "pages": "80--92",
        "other_ids": {
            "doi": [
                "10.1109/TWC.2014.2360196"
            ],

        }
    }

    """

    def __init__(
        self,
        bib_id: str,
        title: str,
        authors: List[Dict[str, str]],
        ref_id: Optional[str] = None,
        year: Optional[int] = None,
        venue: Optional[str] = None,
        volume: Optional[str] = None,
        issue: Optional[str] = None,
        pages: Optional[str] = None,
        other_ids: Dict[str, List] = None,
        num: Optional[int] = None,
        urls: Optional[List] = None,
        raw_text: Optional[str] = None,
        links: Optional[List] = None,
    ):
        self.bib_id = bib_id
        self.ref_id = ref_id
        self.title = title
        self.authors = authors
        self.year = year
        self.venue = venue
        self.volume = volume
        self.issue = issue
        self.pages = pages
        self.other_ids = other_ids
        self.num = num
        self.urls = urls
        self.raw_text = raw_text
        self.links = links

    def as_json(self):
        return {
            "ref_id": self.ref_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "other_ids": self.other_ids,
            "num": self.num,
            "urls": self.urls,
            "raw_text": self.raw_text,
            "links": self.links,
        }


class Affiliation:
    """
    Class for representing affiliation info

    Example:
        {
            "laboratory": "Key Laboratory of Urban Environment and Health",
            "institution": "Chinese Academy of Sciences",
            "location": {
              "postCode": "361021",
              "settlement": "Xiamen",
              "country": "People's Republic of China"
        }
    """

    def __init__(self, laboratory: str, institution: str, location: Dict):
        self.laboratory = laboratory
        self.institution = institution
        self.location = location

    def as_json(self):
        return {
            "laboratory": self.laboratory,
            "institution": self.institution,
            "location": self.location,
        }


class Author:
    """
    Class for representing paper authors

    Example:

        {
          "first": "Anyi",
          "middle": [],
          "last": "Hu",
          "suffix": "",
          "affiliation": {
            "laboratory": "Key Laboratory of Urban Environment and Health",
            "institution": "Chinese Academy of Sciences",
            "location": {
              "postCode": "361021",
              "settlement": "Xiamen",
              "country": "People's Republic of China"
            }
          },
          "email": ""
        }
    """

    def __init__(
        self,
        first: str,
        middle: List[str],
        last: str,
        suffix: str,
        affiliation: Optional[Dict] = None,
        email: Optional[str] = None,
    ):
        self.first = first
        self.middle = middle
        self.last = last
        self.suffix = suffix
        self.affiliation = Affiliation(**affiliation) if affiliation else {}
        self.email = email

    def as_json(self):
        return {
            "first": self.first,
            "middle": self.middle,
            "last": self.last,
            "suffix": self.suffix,
            "affiliation": self.affiliation.as_json() if self.affiliation else {},
            "email": self.email,
        }


class Metadata:
    """
    Class for representing paper metadata

    Example:
    {
      "title": "Niche Partitioning...",
      "authors": [
        {
          "first": "Anyi",
          "middle": [],
          "last": "Hu",
          "suffix": "",
          "affiliation": {
            "laboratory": "Key Laboratory of Urban Environment and Health",
            "institution": "Chinese Academy of Sciences",
            "location": {
              "postCode": "361021",
              "settlement": "Xiamen",
              "country": "People's Republic of China"
            }
          },
          "email": ""
        }
      ],
      "year": "2011-11"
    }
    """

    def __init__(
        self,
        title: str,
        authors: List[Dict],
        year: Optional[str] = None,
        venue: Optional[str] = None,
        identifiers: Optional[Dict] = {},
    ):
        self.title = title
        self.authors = [Author(**author) for author in authors]
        self.year = year
        self.venue = venue
        self.identifiers = identifiers

    def as_json(self):
        return {
            "title": self.title,
            "authors": [author.as_json() for author in self.authors],
            "year": self.year,
            "venue": self.venue,
            "identifiers": self.identifiers,
        }


class Paragraph:
    """
    Class for representing a parsed paragraph from Grobid xml
    All xml tags are removed from the paragraph text, all figures, equations, and tables are replaced
    with a special token that maps to a reference identifier
    Citation mention spans and section header are extracted

    An example json representation (values are examples, not accurate):

    {
        "text": "Formal language techniques BID1 may be used to study FORMULA0 (see REF0)...",
        "mention_spans": [
            {
                "start": 27,
                "end": 31,
                "text": "[1]")
        ],
        "ref_spans": [
            {
                "start": ,
                "end": ,
                "text": "Fig. 1"
            }
        ],
        "eq_spans": [
            {
                "start": 53,
                "end": 61,
                "text": "α = 1",
                "latex": "\\alpha = 1",
                "ref_id": null
            }
        ],
        "section": "Abstract"
    }
    """

    def __init__(
        self,
        text: str,
        cite_spans: List[Dict],
        ref_spans: List[Dict],
        eq_spans: Optional[List[Dict]] = [],
        section: Optional = None,
        sec_num: Optional = None,
    ):
        self.text = text
        self.cite_spans = cite_spans
        self.ref_spans = ref_spans
        self.eq_spans = eq_spans
        if type(section) == str:
            if section:
                sec_parts = section.split("::")
                section_list = [[None, sec_name] for sec_name in sec_parts]
            else:
                section_list = None
            if section_list and sec_num:
                section_list[-1][0] = sec_num
        else:
            section_list = section
        self.section = section_list

    def as_json(self):
        return {
            "text": self.text,
            "cite_spans": self.cite_spans,
            "ref_spans": self.ref_spans,
            "eq_spans": self.eq_spans,
            "section": "::".join([sec[1] for sec in self.section])
            if self.section
            else "",
            "sec_num": self.section[-1][0] if self.section else None,
        }


class Paper:
    """
    Class for representing a parsed S2ORC paper
    """

    def __init__(
        self,
        paper_id: str,
        pdf_hash: str,
        metadata: Dict,
        abstract: List[Dict],
        body_text: List[Dict],
        back_matter: List[Dict],
        bib_entries: Dict,
        ref_entries: Dict,
    ):
        self.paper_id = paper_id
        self.pdf_hash = pdf_hash
        self.metadata = Metadata(**metadata)
        self.abstract = [Paragraph(**para) for para in abstract]
        self.body_text = [Paragraph(**para) for para in body_text]
        self.back_matter = [Paragraph(**para) for para in back_matter]
        self.bib_entries = [
            BibliographyEntry(
                bib_id=key,
                **{
                    CORRECT_KEYS[k] if k in CORRECT_KEYS else k: v
                    for k, v in bib.items()
                    if k not in SKIP_KEYS
                },
            )
            for key, bib in bib_entries.items()
        ]
        self.ref_entries = [
            ReferenceEntry(
                ref_id=key,
                **{
                    CORRECT_KEYS[k] if k in CORRECT_KEYS else k: v
                    for k, v in ref.items()
                    if k != "ref_id"
                },
            )
            for key, ref in ref_entries.items()
        ]

    def as_json(self):
        return {
            "paper_id": self.paper_id,
            "pdf_hash": self.pdf_hash,
            "metadata": self.metadata.as_json(),
            "abstract": [para.as_json() for para in self.abstract],
            "body_text": [para.as_json() for para in self.body_text],
            "back_matter": [para.as_json() for para in self.back_matter],
            "bib_entries": {bib.bib_id: bib.as_json() for bib in self.bib_entries},
            "ref_entries": {ref.ref_id: ref.as_json() for ref in self.ref_entries},
        }

    @property
    def raw_abstract_text(self) -> str:
        """
        Get all the body text joined by a newline
        :return:
        """
        return "\n".join([para.text for para in self.abstract])

    @property
    def raw_body_text(self) -> str:
        """
        Get all the body text joined by a newline
        :return:
        """
        return "\n".join([para.text for para in self.body_text])

    def release_json(self, doc_type: str = "pdf"):
        """
        Return in release JSON format
        :return:
        """
        # TODO: not fully implemented; metadata format is not right; extra keys in some places
        release_dict = {"paper_id": self.paper_id}
        release_dict.update(
            {
                "header": {
                    "generated_with": f"{S2ORC_NAME_STRING} {S2ORC_VERSION_STRING}",
                    "date_generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                }
            }
        )
        release_dict.update(self.metadata.as_json())
        release_dict.update({"abstract": self.raw_abstract_text})
        release_dict.update(
            {
                f"{doc_type}_parse": {
                    "paper_id": self.paper_id,
                    "_pdf_hash": self.pdf_hash,
                    "abstract": [para.as_json() for para in self.abstract],
                    "body_text": [para.as_json() for para in self.body_text],
                    "back_matter": [para.as_json() for para in self.back_matter],
                    "bib_entries": {
                        bib.bib_id: bib.as_json() for bib in self.bib_entries
                    },
                    "ref_entries": {
                        ref.ref_id: ref.as_json() for ref in self.ref_entries
                    },
                }
            }
        )
        return release_dict


def load_s2orc(paper_dict: Dict) -> Paper:
    """
    Load release S2ORC into Paper class
    :param paper_dict:
    :return:
    """
    paper_id = paper_dict["paper_id"]
    pdf_hash = paper_dict.get("_pdf_hash", paper_dict.get("s2_pdf_hash", None))

    # 2019 gorc parses
    if "grobid_parse" in paper_dict and paper_dict.get("grobid_parse"):
        metadata = {
            k: v for k, v in paper_dict["metadata"].items() if k in METADATA_KEYS
        }
        abstract = paper_dict.get("grobid_parse").get("abstract", [])
        body_text = paper_dict.get("grobid_parse").get("body_text", [])
        back_matter = paper_dict.get("grobid_parse").get("back_matter", [])
        bib_entries = paper_dict.get("grobid_parse").get("bib_entries", {})
        for k, v in bib_entries.items():
            if "link" in v:
                v["links"] = [v["link"]]
        ref_entries = paper_dict.get("grobid_parse").get("ref_entries", {})
    # current and 2020 s2orc release_json
    elif ("pdf_parse" in paper_dict and paper_dict.get("pdf_parse")) or (
        "body_text" in paper_dict and paper_dict.get("body_text")
    ):
        if "pdf_parse" in paper_dict:
            paper_dict = paper_dict["pdf_parse"]
        if paper_dict.get("metadata"):
            metadata = {
                k: v
                for k, v in paper_dict.get("metadata").items()
                if k in METADATA_KEYS
            }
        # 2020 s2orc releases (metadata is separate)
        else:
            metadata = {"title": None, "authors": [], "year": None}
        abstract = paper_dict.get("abstract", [])
        body_text = paper_dict.get("body_text", [])
        back_matter = paper_dict.get("back_matter", [])
        bib_entries = paper_dict.get("bib_entries", {})
        for k, v in bib_entries.items():
            if "link" in v:
                v["links"] = [v["link"]]
        ref_entries = paper_dict.get("ref_entries", {})
    else:
        print(paper_id)
        raise NotImplementedError("Unknown S2ORC file type!")

    return Paper(
        paper_id=paper_id,
        pdf_hash=pdf_hash,
        metadata=metadata,
        abstract=abstract,
        body_text=body_text,
        back_matter=back_matter,
        bib_entries=bib_entries,
        ref_entries=ref_entries,
    )


BRACKET_REGEX = re.compile(r"\[[1-9]\d{0,2}([,;\-\s]+[1-9]\d{0,2})*;?\]")
BRACKET_STYLE_THRESHOLD = 5

SINGLE_BRACKET_REGEX = re.compile(r"\[([1-9]\d{0,2})\]")
EXPANSION_CHARS = {"-", "–"}

REPLACE_TABLE_TOKS = {
    "<row>": "<tr>",
    "<row/>": "<tr/>",
    "</row>": "</tr>",
    "<cell>": "<td>",
    "<cell/>": "<td/>",
    "</cell>": "</td>",
    "<cell ": "<td ",
    "cols=": "colspan=",
}


def span_already_added(
    sub_start: int, sub_end: int, span_indices: List[Tuple[int, int]]
) -> bool:
    """
    Check if span is a subspan of existing span
    :param sub_start:
    :param sub_end:
    :param span_indices:
    :return:
    """
    for span_start, span_end in span_indices:
        if sub_start >= span_start and sub_end <= span_end:
            return True
    return False


def is_expansion_string(between_string: str) -> bool:
    """
    Check if the string between two refs is an expansion string
    :param between_string:
    :return:
    """
    if (
        len(between_string) <= 2
        and any([c in EXPANSION_CHARS for c in between_string])
        and all([c in EXPANSION_CHARS.union({" "}) for c in between_string])
    ):
        return True
    return False


# TODO: still cases like `09bcee03baceb509d4fcf736fa1322cb8adf507f` w/ dups like ['L Jung', 'R Hessler', 'Louis Jung', 'Roland Hessler']
# example paper that has empties & duplicates: `09bce26cc7e825e15a4469e3e78b7a54898bb97f`
def _clean_empty_and_duplicate_authors_from_grobid_parse(
    authors: List[Dict],
) -> List[Dict]:
    """
    Within affiliation, `location` is a dict with fields <settlement>, <region>, <country>, <postCode>, etc.
    Too much hassle, so just take the first one that's not empty.
    """
    # stripping empties
    clean_authors_list = []
    for author in authors:
        clean_first = author["first"].strip()
        clean_last = author["last"].strip()
        clean_middle = [m.strip() for m in author["middle"]]
        clean_suffix = author["suffix"].strip()
        if clean_first or clean_last or clean_middle:
            author["first"] = clean_first
            author["last"] = clean_last
            author["middle"] = clean_middle
            author["suffix"] = clean_suffix
            clean_authors_list.append(author)
    # combining duplicates (preserve first occurrence of author name as position)
    key_to_author_blobs = {}
    ordered_keys_by_author_pos = []
    for author in clean_authors_list:
        key = (
            author["first"],
            author["last"],
            " ".join(author["middle"]),
            author["suffix"],
        )
        if key not in key_to_author_blobs:
            key_to_author_blobs[key] = author
            ordered_keys_by_author_pos.append(key)
        else:
            if author["email"]:
                key_to_author_blobs[key]["email"] = author["email"]
            if author["affiliation"] and (
                author["affiliation"]["institution"]
                or author["affiliation"]["laboratory"]
                or author["affiliation"]["location"]
            ):
                key_to_author_blobs[key]["affiliation"] = author["affiliation"]
    dedup_authors_list = [
        key_to_author_blobs[key] for key in ordered_keys_by_author_pos
    ]
    return dedup_authors_list


def sub_spans_and_update_indices(
    spans_to_replace: List[Tuple[int, int, str, str]], full_string: str
) -> Tuple[str, List]:
    """
    Replace all spans and recompute indices
    :param spans_to_replace:
    :param full_string:
    :return:
    """
    # TODO: check no spans overlapping
    # TODO: check all spans well-formed

    # assert all spans are equal to full_text span
    assert all(
        [full_string[start:end] == token for start, end, token, _ in spans_to_replace]
    )

    # assert none of the spans start with the same start ind
    start_inds = [rep[0] for rep in spans_to_replace]
    assert len(set(start_inds)) == len(start_inds)

    # sort by start index
    spans_to_replace.sort(key=lambda x: x[0])

    # compute offsets for each span
    new_spans = [
        [start, end, token, surface, 0]
        for start, end, token, surface in spans_to_replace
    ]
    for i, entry in enumerate(spans_to_replace):
        start, end, token, surface = entry
        new_end = start + len(surface)
        offset = new_end - end
        new_spans[i][1] += offset
        for new_span_entry in new_spans[i + 1 :]:
            new_span_entry[4] += offset

    # generate new text and create final spans
    new_text = replace_refspans(spans_to_replace, full_string, btwn_padding="")
    new_spans = [
        (start + offset, end + offset, token, surface)
        for start, end, token, surface, offset in new_spans
    ]

    return new_text, new_spans


def extract_paper_metadata_from_grobid_xml(tag: bs4.element.Tag) -> Dict:
    """
    Extract paper metadata (title, authors, affiliation, year) from grobid xml
    :param tag:
    :return:
    """
    clean_tags(tag)
    paper_metadata = {
        "title": tag.titlestmt.title.text,
        "authors": get_author_data_from_grobid_xml(tag),
        "year": get_publication_datetime_from_grobid_xml(tag),
    }
    return paper_metadata


def parse_bib_entry(bib_entry: BeautifulSoup) -> Dict:
    """
    Parse one bib entry
    :param bib_entry:
    :return:
    """
    clean_tags(bib_entry)
    title = get_title_from_grobid_xml(bib_entry)
    return {
        "ref_id": bib_entry.attrs.get("xml:id", None),
        "title": title,
        "authors": get_author_names_from_grobid_xml(bib_entry),
        "year": get_year_from_grobid_xml(bib_entry),
        "venue": get_venue_from_grobid_xml(bib_entry, title),
        "volume": get_volume_from_grobid_xml(bib_entry),
        "issue": get_issue_from_grobid_xml(bib_entry),
        "pages": get_pages_from_grobid_xml(bib_entry),
        "other_ids": get_other_ids_from_grobid_xml(bib_entry),
        "raw_text": get_raw_bib_text_from_grobid_xml(bib_entry),
        "urls": [],
    }


"""
This version uses the standard ProcessPoolExecutor for parallelizing the concurrent calls to the GROBID services.
Given the limits of ThreadPoolExecutor (input stored in memory, blocking Executor.map until the whole input
is acquired), it works with batches of PDF of a size indicated in the config.json file (default is 1000 entries).
We are moving from first batch to the second one only when the first is entirely processed - which means it is
slightly sub-optimal, but should scale better. However acquiring a list of million of files in directories would
require something scalable too, which is not implemented for the moment.
"""

DEFAULT_GROBID_CONFIG = {
    "grobid_server": "localhost",
    "grobid_port": "8070",
    "batch_size": 1000,
    "sleep_time": 5,
    "generateIDs": False,
    "consolidate_header": False,
    "consolidate_citations": False,
    "include_raw_citations": True,
    "include_raw_affiliations": False,
    "max_workers": 2,
}


class GrobidClient(ApiClient):
    def __init__(self, config=None):
        self.config = config or DEFAULT_GROBID_CONFIG
        self.generate_ids = self.config["generateIDs"]
        self.consolidate_header = self.config["consolidate_header"]
        self.consolidate_citations = self.config["consolidate_citations"]
        self.include_raw_citations = self.config["include_raw_citations"]
        self.include_raw_affiliations = self.config["include_raw_affiliations"]
        self.max_workers = self.config["max_workers"]
        self.grobid_server = self.config["grobid_server"]
        self.grobid_port = self.config["grobid_port"]
        self.sleep_time = self.config["sleep_time"]

    def process(self, input: str, output: str, service: str):
        batch_size_pdf = self.config["batch_size"]
        pdf_files = []

        for pdf_file in glob.glob(input + "/*.pdf"):
            pdf_files.append(pdf_file)

            if len(pdf_files) == batch_size_pdf:
                self.process_batch(pdf_files, output, service)
                pdf_files = []

        # last batch
        if len(pdf_files) > 0:
            self.process_batch(pdf_files, output, service)

    def process_batch(self, pdf_files: List[str], output: str, service: str) -> None:
        print(len(pdf_files), "PDF files to process")
        for pdf_file in pdf_files:
            self.process_pdf(pdf_file, output, service)

    def process_pdf_stream(
        self, pdf_file: str, pdf_strm: bytes, output: str, service: str
    ) -> str:
        # process the stream
        files = {"input": (pdf_file, pdf_strm, "application/pdf", {"Expires": "0"})}

        the_url = "http://" + self.grobid_server
        the_url += ":" + self.grobid_port
        the_url += "/api/" + service

        # set the GROBID parameters
        the_data = {}
        if self.generate_ids:
            the_data["generateIDs"] = "1"
        else:
            the_data["generateIDs"] = "0"

        if self.consolidate_header:
            the_data["consolidateHeader"] = "1"
        else:
            the_data["consolidateHeader"] = "0"

        if self.consolidate_citations:
            the_data["consolidateCitations"] = "1"
        else:
            the_data["consolidateCitations"] = "0"

        if self.include_raw_affiliations:
            the_data["includeRawAffiliations"] = "1"
        else:
            the_data["includeRawAffiliations"] = "0"

        if self.include_raw_citations:
            the_data["includeRawCitations"] = "1"
        else:
            the_data["includeRawCitations"] = "0"

        res, status = self.post(
            url=the_url, files=files, data=the_data, headers={"Accept": "text/plain"}
        )

        if status == 503:
            time.sleep(self.sleep_time)
            return self.process_pdf_stream(pdf_file, pdf_strm, service)
        elif status != 200:
            with open(os.path.join(output, "failed.log"), "a+") as failed:
                failed.write(pdf_file.strip(".pdf") + "\n")
            print("Processing failed with error " + str(status))
            return ""
        else:
            return res.text

    def process_pdf(self, pdf_file: str, output: str, service: str) -> None:
        # check if TEI file is already produced
        # we use ntpath here to be sure it will work on Windows too
        pdf_file_name = ntpath.basename(pdf_file)
        filename = os.path.join(output, os.path.splitext(pdf_file_name)[0] + ".tei.xml")
        if os.path.isfile(filename):
            return

        print(pdf_file)
        pdf_strm = open(pdf_file, "rb").read()
        tei_text = self.process_pdf_stream(pdf_file, pdf_strm, output, service)

        # writing TEI file
        if tei_text:
            with io.open(filename, "w+", encoding="utf8") as tei_file:
                tei_file.write(tei_text)

    def process_citation(self, bib_string: str, log_file: str) -> str:
        # process citation raw string and return corresponding dict
        the_data = {"citations": bib_string, "consolidateCitations": "0"}

        the_url = "http://" + self.grobid_server
        the_url += ":" + self.grobid_port
        the_url += "/api/processCitation"

        for _ in range(5):
            try:
                res, status = self.post(
                    url=the_url, data=the_data, headers={"Accept": "text/plain"}
                )
                if status == 503:
                    time.sleep(self.sleep_time)
                    continue
                elif status != 200:
                    with open(log_file, "a+") as failed:
                        failed.write("-- BIBSTR --\n")
                        failed.write(bib_string + "\n\n")
                    break
                else:
                    return res.text
            except Exception:
                continue

    def process_header_names(self, header_string: str, log_file: str) -> str:
        # process author names from header string
        the_data = {"names": header_string}

        the_url = "http://" + self.grobid_server
        the_url += ":" + self.grobid_port
        the_url += "/api/processHeaderNames"

        res, status = self.post(
            url=the_url, data=the_data, headers={"Accept": "text/plain"}
        )

        if status == 503:
            time.sleep(self.sleep_time)
            return self.process_header_names(header_string, log_file)
        elif status != 200:
            with open(log_file, "a+") as failed:
                failed.write("-- AUTHOR --\n")
                failed.write(header_string + "\n\n")
        else:
            return res.text

    def process_affiliations(self, aff_string: str, log_file: str) -> str:
        # process affiliation from input string
        the_data = {"affiliations": aff_string}

        the_url = "http://" + self.grobid_server
        the_url += ":" + self.grobid_port
        the_url += "/api/processAffiliations"

        res, status = self.post(
            url=the_url, data=the_data, headers={"Accept": "text/plain"}
        )

        if status == 503:
            time.sleep(self.sleep_time)
            return self.process_affiliations(aff_string, log_file)
        elif status != 200:
            with open(log_file, "a+") as failed:
                failed.write("-- AFFILIATION --\n")
                failed.write(aff_string + "\n\n")
        else:
            return res.text


class UniqTokenGenerator:
    """
    Generate unique token
    """

    def __init__(self, tok_string):
        self.tok_string = tok_string
        self.ind = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        new_token = f"{self.tok_string}{self.ind}"
        self.ind += 1
        return new_token


def normalize_grobid_id(grobid_id: str):
    """
    Normalize grobid object identifiers
    :param grobid_id:
    :return:
    """
    str_norm = grobid_id.upper().replace("_", "").replace("#", "")
    if str_norm.startswith("B"):
        return str_norm.replace("B", "BIBREF")
    if str_norm.startswith("TAB"):
        return str_norm.replace("TAB", "TABREF")
    if str_norm.startswith("FIG"):
        return str_norm.replace("FIG", "FIGREF")
    if str_norm.startswith("FORMULA"):
        return str_norm.replace("FORMULA", "EQREF")
    return str_norm


def parse_bibliography(soup: BeautifulSoup) -> List[Dict]:
    """
    Finds all bibliography entries in a grobid xml.
    """
    bibliography = soup.listBibl
    if bibliography is None:
        return []

    entries = bibliography.find_all("biblStruct")

    structured_entries = []
    for entry in entries:
        bib_entry = parse_bib_entry(entry)
        # add bib entry only if it has a title
        if bib_entry["title"]:
            structured_entries.append(bib_entry)

    bibliography.decompose()

    return structured_entries


def extract_formulas_from_tei_xml(sp: BeautifulSoup) -> None:
    """
    Replace all formulas with the text
    :param sp:
    :return:
    """
    for eq in sp.find_all("formula"):
        eq.replace_with(sp.new_string(eq.text.strip()))


def table_to_html(table: bs4.element.Tag) -> str:
    """
    Sub table tags with html table tags
    :param table_str:
    :return:
    """
    for tag in table:
        if tag.name != "row":
            print(f"Unknown table subtag: {tag.name}")
            tag.decompose()
    table_str = str(table)
    for token, subtoken in REPLACE_TABLE_TOKS.items():
        table_str = table_str.replace(token, subtoken)
    return table_str


def extract_figures_and_tables_from_tei_xml(sp: BeautifulSoup) -> Dict[str, Dict]:
    """
    Generate figure and table dicts
    :param sp:
    :return:
    """
    ref_map = dict()

    for fig in sp.find_all("figure"):
        try:
            if fig.name and fig.get("xml:id"):
                if fig.get("type") == "table":
                    ref_map[normalize_grobid_id(fig.get("xml:id"))] = {
                        "text": fig.figDesc.text.strip()
                        if fig.figDesc
                        else fig.head.text.strip()
                        if fig.head
                        else "",
                        "latex": None,
                        "type": "table",
                        "content": table_to_html(fig.table),
                        "fig_num": fig.get("xml:id"),
                    }
                else:
                    if True in [
                        char.isdigit()
                        for char in fig.findNext("head").findNext("label")
                    ]:
                        fig_num = fig.findNext("head").findNext("label").contents[0]
                    else:
                        fig_num = None
                    ref_map[normalize_grobid_id(fig.get("xml:id"))] = {
                        "text": fig.figDesc.text.strip() if fig.figDesc else "",
                        "latex": None,
                        "type": "figure",
                        "content": "",
                        "fig_num": fig_num,
                    }
        except AttributeError:
            continue
        fig.decompose()

    return ref_map


def check_if_citations_are_bracket_style(sp: BeautifulSoup) -> bool:
    """
    Check if the document has bracket style citations
    :param sp:
    :return:
    """
    cite_strings = []
    if sp.body:
        for div in sp.body.find_all("div"):
            if div.head:
                continue
            for rtag in div.find_all("ref"):
                ref_type = rtag.get("type")
                if ref_type == "bibr":
                    cite_strings.append(rtag.text.strip())

        # check how many match bracket style
        bracket_style = [
            bool(BRACKET_REGEX.match(cite_str)) for cite_str in cite_strings
        ]

        # return true if
        if sum(bracket_style) > BRACKET_STYLE_THRESHOLD:
            return True

    return False


def sub_all_note_tags(sp: BeautifulSoup) -> BeautifulSoup:
    """
    Sub all note tags with p tags
    :param para_el:
    :param sp:
    :return:
    """
    for ntag in sp.find_all("note"):
        p_tag = sp.new_tag("p")
        p_tag.string = ntag.text.strip()
        ntag.replace_with(p_tag)
    return sp


def process_formulas_in_paragraph(para_el: BeautifulSoup, sp: BeautifulSoup) -> None:
    """
    Process all formulas in paragraph and replace with text and label
    :param para_el:
    :param sp:
    :return:
    """
    for ftag in para_el.find_all("formula"):
        # get label if exists and insert a space between formula and label
        if ftag.label:
            label = " " + ftag.label.text
            ftag.label.decompose()
        else:
            label = ""
        ftag.replace_with(sp.new_string(f"{ftag.text.strip()}{label}"))


def process_references_in_paragraph(
    para_el: BeautifulSoup, sp: BeautifulSoup, refs: Dict
) -> Dict:
    """
    Process all references in paragraph and generate a dict that contains (type, ref_id, surface_form)
    :param para_el:
    :param sp:
    :param refs:
    :return:
    """
    tokgen = UniqTokenGenerator("REFTOKEN")
    ref_dict = dict()
    for rtag in para_el.find_all("ref"):
        try:
            ref_type = rtag.get("type")
            # skip if citation
            if ref_type == "bibr":
                continue
            if ref_type == "table" or ref_type == "figure":
                ref_id = rtag.get("target")
                if ref_id and normalize_grobid_id(ref_id) in refs:
                    # normalize reference string
                    rtag_string = normalize_grobid_id(ref_id)
                else:
                    rtag_string = None
                # add to ref set
                ref_key = tokgen.next()
                ref_dict[ref_key] = (rtag_string, rtag.text.strip(), ref_type)
                rtag.replace_with(sp.new_string(f" {ref_key} "))
            else:
                # replace with surface form
                rtag.replace_with(sp.new_string(rtag.text.strip()))
        except AttributeError:
            continue
    return ref_dict


def process_citations_in_paragraph(
    para_el: BeautifulSoup, sp: BeautifulSoup, bibs: Dict, bracket: bool
) -> Dict:
    """
    Process all citations in paragraph and generate a dict for surface forms
    :param para_el:
    :param sp:
    :param bibs:
    :param bracket:
    :return:
    """

    # CHECK if range between two surface forms is appropriate for bracket style expansion
    def _get_surface_range(start_surface, end_surface):
        span1_match = SINGLE_BRACKET_REGEX.match(start_surface)
        span2_match = SINGLE_BRACKET_REGEX.match(end_surface)
        if span1_match and span2_match:
            # get numbers corresponding to citations
            span1_num = int(span1_match.group(1))
            span2_num = int(span2_match.group(1))
            # expand if range is between 1 and 20
            if 1 < span2_num - span1_num < 20:
                return span1_num, span2_num
        return None

    # CREATE BIBREF range between two reference ids, e.g. BIBREF1-BIBREF4 -> BIBREF1 BIBREF2 BIBREF3 BIBREF4
    def _create_ref_id_range(start_ref_id, end_ref_id):
        start_ref_num = int(start_ref_id[6:])
        end_ref_num = int(end_ref_id[6:])
        return [
            f"BIBREF{curr_ref_num}"
            for curr_ref_num in range(start_ref_num, end_ref_num + 1)
        ]

    # CREATE surface form range between two bracket strings, e.g. [1]-[4] -> [1] [2] [3] [4]
    def _create_surface_range(start_number, end_number):
        return [f"[{n}]" for n in range(start_number, end_number + 1)]

    # create citation dict with keywords
    cite_map = dict()
    tokgen = UniqTokenGenerator("CITETOKEN")

    for rtag in para_el.find_all("ref"):
        try:
            # get surface span, e.g. [3]
            surface_span = rtag.text.strip()

            # check if target is available (#b2 -> BID2)
            if rtag.get("target"):
                # normalize reference string
                rtag_ref_id = normalize_grobid_id(rtag.get("target"))

                # skip if rtag ref_id not in bibliography
                if rtag_ref_id not in bibs:
                    cite_key = tokgen.next()
                    rtag.replace_with(sp.new_string(f" {cite_key} "))
                    cite_map[cite_key] = (None, surface_span)
                    continue

                # if bracket style, only keep if surface form is bracket
                if bracket:
                    # valid bracket span
                    if surface_span and (
                        surface_span[0] == "["
                        or surface_span[-1] == "]"
                        or surface_span[-1] == ","
                    ):
                        pass
                    # invalid, replace tag with surface form and continue to next ref tag
                    else:
                        rtag.replace_with(sp.new_string(f" {surface_span} "))
                        continue
                # not bracket, add cite span and move on
                else:
                    cite_key = tokgen.next()
                    rtag.replace_with(sp.new_string(f" {cite_key} "))
                    cite_map[cite_key] = (rtag_ref_id, surface_span)
                    continue

                ### EXTRA PROCESSING FOR BRACKET STYLE CITATIONS; EXPAND RANGES ###
                # look backward for range marker, e.g. [1]-*[3]*
                backward_between_span = ""
                for sib in rtag.previous_siblings:
                    if sib.name == "ref":
                        break
                    elif type(sib) == NavigableString:
                        backward_between_span += sib
                    else:
                        break

                # check if there's a backwards expansion, e.g. need to expand [1]-[3] -> [1] [2] [3]
                if is_expansion_string(backward_between_span):
                    # get surface number range
                    surface_num_range = _get_surface_range(
                        rtag.find_previous_sibling("ref").text.strip(), surface_span
                    )
                    # if the surface number range is reasonable (range < 20, in order), EXPAND
                    if surface_num_range:
                        # delete previous ref tag and anything in between (i.e. delete "-" and extra spaces)
                        for sib in rtag.previous_siblings:
                            if sib.name == "ref":
                                break
                            elif type(sib) == NavigableString:
                                sib.replace_with(sp.new_string(""))
                            else:
                                break

                        # get ref id of previous ref, e.g. [1] (#b0 -> BID0)
                        previous_rtag = rtag.find_previous_sibling("ref")
                        previous_rtag_ref_id = normalize_grobid_id(
                            previous_rtag.get("target")
                        )
                        previous_rtag.decompose()

                        # replace this ref tag with the full range expansion, e.g. [3] (#b2 -> BID1 BID2)
                        id_range = _create_ref_id_range(
                            previous_rtag_ref_id, rtag_ref_id
                        )
                        surface_range = _create_surface_range(
                            surface_num_range[0], surface_num_range[1]
                        )
                        replace_string = ""
                        for range_ref_id, range_surface_form in zip(
                            id_range, surface_range
                        ):
                            # only replace if ref id is in bibliography, else add none
                            if range_ref_id in bibs:
                                cite_key = tokgen.next()
                                cite_map[cite_key] = (range_ref_id, range_surface_form)
                            else:
                                cite_key = tokgen.next()
                                cite_map[cite_key] = (None, range_surface_form)
                            replace_string += cite_key + " "
                        rtag.replace_with(sp.new_string(f" {replace_string} "))
                    # ELSE do not expand backwards and replace previous and current rtag with appropriate ref id
                    else:
                        # add mapping between ref id and surface form for previous ref tag
                        previous_rtag = rtag.find_previous_sibling("ref")
                        previous_rtag_ref_id = normalize_grobid_id(
                            previous_rtag.get("target")
                        )
                        previous_rtag_surface = previous_rtag.text.strip()
                        cite_key = tokgen.next()
                        previous_rtag.replace_with(sp.new_string(f" {cite_key} "))
                        cite_map[cite_key] = (
                            previous_rtag_ref_id,
                            previous_rtag_surface,
                        )

                        # add mapping between ref id and surface form for current reftag
                        cite_key = tokgen.next()
                        rtag.replace_with(sp.new_string(f" {cite_key} "))
                        cite_map[cite_key] = (rtag_ref_id, surface_span)
                else:
                    # look forward and see if expansion string, e.g. *[1]*-[3]
                    forward_between_span = ""
                    for sib in rtag.next_siblings:
                        if sib.name == "ref":
                            break
                        elif type(sib) == NavigableString:
                            forward_between_span += sib
                        else:
                            break
                    # look forward for range marker (if is a range, continue -- range will be expanded
                    # when we get to the second value)
                    if is_expansion_string(forward_between_span):
                        continue
                    # else treat like normal reference
                    else:
                        cite_key = tokgen.next()
                        rtag.replace_with(sp.new_string(f" {cite_key} "))
                        cite_map[cite_key] = (rtag_ref_id, surface_span)

            else:
                cite_key = tokgen.next()
                rtag.replace_with(sp.new_string(f" {cite_key} "))
                cite_map[cite_key] = (None, surface_span)
        except AttributeError:
            continue

    return cite_map


def process_paragraph(
    sp: BeautifulSoup,
    para_el: bs4.element.Tag,
    section_names: List[Tuple],
    bib_dict: Dict,
    ref_dict: Dict,
    bracket: bool,
) -> Dict:
    """
    Process one paragraph
    :param sp:
    :param para_el:
    :param section_names:
    :param bib_dict:
    :param ref_dict:
    :param bracket: if bracket style, expand and clean up citations
    :return:
    """
    # return empty paragraph if no text
    if not para_el.text:
        return {
            "text": "",
            "cite_spans": [],
            "ref_spans": [],
            "eq_spans": [],
            "section": section_names,
        }

    # replace formulas with formula text
    process_formulas_in_paragraph(para_el, sp)

    # get references to tables and figures
    ref_map = process_references_in_paragraph(para_el, sp, ref_dict)

    # generate citation map for paragraph element (keep only cite spans with bib entry or unlinked)
    cite_map = process_citations_in_paragraph(para_el, sp, bib_dict, bracket)

    # substitute space characters
    para_text = re.sub(r"\s+", " ", para_el.text)
    para_text = re.sub(r"\s", " ", para_text)

    # get all cite and ref spans
    all_spans_to_replace = []
    for span in re.finditer(r"(CITETOKEN\d+)", para_text):
        uniq_token = span.group()
        ref_id, surface_text = cite_map[uniq_token]
        all_spans_to_replace.append(
            (span.start(), span.start() + len(uniq_token), uniq_token, surface_text)
        )
    for span in re.finditer(r"(REFTOKEN\d+)", para_text):
        uniq_token = span.group()
        ref_id, surface_text, ref_type = ref_map[uniq_token]
        all_spans_to_replace.append(
            (span.start(), span.start() + len(uniq_token), uniq_token, surface_text)
        )

    # replace cite and ref spans and create json blobs
    para_text, all_spans_to_replace = sub_spans_and_update_indices(
        all_spans_to_replace, para_text
    )

    cite_span_blobs = [
        {"start": start, "end": end, "text": surface, "ref_id": cite_map[token][0]}
        for start, end, token, surface in all_spans_to_replace
        if token.startswith("CITETOKEN")
    ]

    ref_span_blobs = [
        {"start": start, "end": end, "text": surface, "ref_id": ref_map[token][0]}
        for start, end, token, surface in all_spans_to_replace
        if token.startswith("REFTOKEN")
    ]

    for cite_blob in cite_span_blobs:
        assert para_text[cite_blob["start"] : cite_blob["end"]] == cite_blob["text"]

    for ref_blob in ref_span_blobs:
        assert para_text[ref_blob["start"] : ref_blob["end"]] == ref_blob["text"]

    return {
        "text": para_text,
        "cite_spans": cite_span_blobs,
        "ref_spans": ref_span_blobs,
        "eq_spans": [],
        "section": section_names,
    }


def extract_abstract_from_tei_xml(
    sp: BeautifulSoup, bib_dict: Dict, ref_dict: Dict, cleanup_bracket: bool
) -> List[Dict]:
    """
    Parse abstract from soup
    :param sp:
    :param bib_dict:
    :param ref_dict:
    :param cleanup_bracket:
    :return:
    """
    abstract_text = []
    if sp.abstract:
        # process all divs
        if sp.abstract.div:
            for div in sp.abstract.find_all("div"):
                if div.text:
                    if div.p:
                        for para in div.find_all("p"):
                            if para.text:
                                abstract_text.append(
                                    process_paragraph(
                                        sp,
                                        para,
                                        [(None, "Abstract")],
                                        bib_dict,
                                        ref_dict,
                                        cleanup_bracket,
                                    )
                                )
                    else:
                        if div.text:
                            abstract_text.append(
                                process_paragraph(
                                    sp,
                                    div,
                                    [(None, "Abstract")],
                                    bib_dict,
                                    ref_dict,
                                    cleanup_bracket,
                                )
                            )
        # process all paragraphs
        elif sp.abstract.p:
            for para in sp.abstract.find_all("p"):
                if para.text:
                    abstract_text.append(
                        process_paragraph(
                            sp,
                            para,
                            [(None, "Abstract")],
                            bib_dict,
                            ref_dict,
                            cleanup_bracket,
                        )
                    )
        # else just try to get the text
        else:
            if sp.abstract.text:
                abstract_text.append(
                    process_paragraph(
                        sp,
                        sp.abstract,
                        [(None, "Abstract")],
                        bib_dict,
                        ref_dict,
                        cleanup_bracket,
                    )
                )
        sp.abstract.decompose()
    return abstract_text


def extract_body_text_from_div(
    sp: BeautifulSoup,
    div: bs4.element.Tag,
    sections: List[Tuple],
    bib_dict: Dict,
    ref_dict: Dict,
    cleanup_bracket: bool,
) -> List[Dict]:
    """
    Parse body text from soup
    :param sp:
    :param div:
    :param sections:
    :param bib_dict:
    :param ref_dict:
    :param cleanup_bracket:
    :return:
    """
    chunks = []
    # check if nested divs; recursively process
    if div.div:
        for subdiv in div.find_all("div"):
            # has header, add to section list and process
            if subdiv.head:
                chunks += extract_body_text_from_div(
                    sp,
                    subdiv,
                    sections + [(subdiv.head.get("n", None), subdiv.head.text.strip())],
                    bib_dict,
                    ref_dict,
                    cleanup_bracket,
                )
                subdiv.head.decompose()
            # no header, process with same section list
            else:
                chunks += extract_body_text_from_div(
                    sp, subdiv, sections, bib_dict, ref_dict, cleanup_bracket
                )
    # process tags individuals
    for tag in div:
        try:
            if tag.name == "p":
                if tag.text:
                    chunks.append(
                        process_paragraph(
                            sp, tag, sections, bib_dict, ref_dict, cleanup_bracket
                        )
                    )
            elif tag.name == "formula":
                # e.g. <formula xml:id="formula_0">Y = W T X.<label>(1)</label></formula>
                label = tag.label.text
                tag.label.decompose()
                eq_text = tag.text
                chunks.append(
                    {
                        "text": "EQUATION",
                        "cite_spans": [],
                        "ref_spans": [],
                        "eq_spans": [
                            {
                                "start": 0,
                                "end": 8,
                                "text": "EQUATION",
                                "ref_id": "EQREF",
                                "raw_str": eq_text,
                                "eq_num": label,
                            }
                        ],
                        "section": sections,
                    }
                )
        except AttributeError:
            if tag.text:
                chunks.append(
                    process_paragraph(
                        sp, tag, sections, bib_dict, ref_dict, cleanup_bracket
                    )
                )

    return chunks


def extract_body_text_from_tei_xml(
    sp: BeautifulSoup, bib_dict: Dict, ref_dict: Dict, cleanup_bracket: bool
) -> List[Dict]:
    """
    Parse body text from soup
    :param sp:
    :param bib_dict:
    :param ref_dict:
    :param cleanup_bracket:
    :return:
    """
    body_text = []
    if sp.body:
        body_text = extract_body_text_from_div(
            sp, sp.body, [], bib_dict, ref_dict, cleanup_bracket
        )
        sp.body.decompose()
    return body_text


def extract_back_matter_from_tei_xml(
    sp: BeautifulSoup, bib_dict: Dict, ref_dict: Dict, cleanup_bracket: bool
) -> List[Dict]:
    """
    Parse back matter from soup
    :param sp:
    :param bib_dict:
    :param ref_dict:
    :param cleanup_bracket:
    :return:
    """
    back_text = []

    if sp.back:
        for div in sp.back.find_all("div"):
            if div.get("type"):
                section_type = div.get("type")
            else:
                section_type = ""

            for child_div in div.find_all("div"):
                if child_div.head:
                    section_title = child_div.head.text.strip()
                    section_num = child_div.head.get("n", None)
                    child_div.head.decompose()
                else:
                    section_title = section_type
                    section_num = None
                if child_div.text:
                    if child_div.text:
                        back_text.append(
                            process_paragraph(
                                sp,
                                child_div,
                                [(section_num, section_title)],
                                bib_dict,
                                ref_dict,
                                cleanup_bracket,
                            )
                        )
        sp.back.decompose()
    return back_text


def convert_tei_xml_soup_to_s2orc_json(
    soup: BeautifulSoup, paper_id: str, pdf_hash: str
) -> Paper:
    """
    Convert Grobid TEI XML to S2ORC json format
    :param soup: BeautifulSoup of XML file content
    :param paper_id: name of file
    :param pdf_hash: hash of PDF
    :return:
    """
    # extract metadata
    metadata = extract_paper_metadata_from_grobid_xml(soup.fileDesc)
    # clean metadata authors (remove dupes etc)
    metadata["authors"] = _clean_empty_and_duplicate_authors_from_grobid_parse(
        metadata["authors"]
    )

    # parse bibliography entries (removes empty bib entries)
    biblio_entries = parse_bibliography(soup)
    bibkey_map = {normalize_grobid_id(bib["ref_id"]): bib for bib in biblio_entries}

    # # process formulas and replace with text
    # extract_formulas_from_tei_xml(soup)

    # extract figure and table captions
    refkey_map = extract_figures_and_tables_from_tei_xml(soup)

    # get bracket style
    is_bracket_style = check_if_citations_are_bracket_style(soup)

    # substitute all note tags with p tags
    soup = sub_all_note_tags(soup)

    # process abstract if possible
    abstract_entries = extract_abstract_from_tei_xml(
        soup, bibkey_map, refkey_map, is_bracket_style
    )

    # process body text
    body_entries = extract_body_text_from_tei_xml(
        soup, bibkey_map, refkey_map, is_bracket_style
    )

    # parse back matter (acks, author statements, competing interests, abbrevs etc)
    back_matter = extract_back_matter_from_tei_xml(
        soup, bibkey_map, refkey_map, is_bracket_style
    )

    # form final paper entry
    return Paper(
        paper_id=paper_id,
        pdf_hash=pdf_hash,
        metadata=metadata,
        abstract=abstract_entries,
        body_text=body_entries,
        back_matter=back_matter,
        bib_entries=bibkey_map,
        ref_entries=refkey_map,
    )


def convert_tei_xml_file_to_s2orc_json(tei_file: str, pdf_hash: str = "") -> Paper:
    """
    Convert a TEI XML file to S2ORC JSON
    :param tei_file:
    :param pdf_hash:
    :return:
    """
    if not os.path.exists(tei_file):
        raise FileNotFoundError("Input TEI XML file doesn't exist")
    paper_id = tei_file.split("/")[-1].split(".")[0]
    soup = BeautifulSoup(open(tei_file, "rb").read(), "xml")
    paper = convert_tei_xml_soup_to_s2orc_json(soup, paper_id, pdf_hash)
    return paper
