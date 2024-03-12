import json
import os
from typing import Dict, Optional

from bs4 import BeautifulSoup

from data_factory.paper2json.doc2json.utils import (
    GrobidClient, convert_tei_xml_file_to_s2orc_json,
    convert_tei_xml_soup_to_s2orc_json)

BASE_TEMP_DIR = "./grobid/temp"
BASE_OUTPUT_DIR = "./grobid/output"
BASE_LOG_DIR = "./grobid/log"


def process_pdf_stream(
    input_file: str, sha: str, input_stream: bytes, grobid_config: Optional[Dict] = None
) -> Dict:
    """
    Process PDF stream
    :param input_file:
    :param sha:
    :param input_stream:
    :return:
    """
    # process PDF through Grobid -> TEI.XML
    client = GrobidClient(grobid_config)
    tei_text = client.process_pdf_stream(
        input_file, input_stream, "temp", "processFulltextDocument"
    )

    # make soup
    soup = BeautifulSoup(tei_text, "xml")

    # get paper
    paper = convert_tei_xml_soup_to_s2orc_json(soup, input_file, sha)

    return paper.release_json("pdf")


def process_pdf_file(
    input_file: str,
    temp_dir: str = BASE_TEMP_DIR,
    output_dir: str = BASE_OUTPUT_DIR,
    grobid_config: Optional[Dict] = None,
) -> str:
    """
    Process a PDF file and get JSON representation
    :param input_file:
    :param temp_dir:
    :param output_dir:
    :return:
    """
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # get paper id as the name of the file
    paper_id = ".".join(input_file.split("/")[-1].split(".")[:-1])
    tei_file = os.path.join(temp_dir, f"{paper_id}.tei.xml")
    output_file = os.path.join(output_dir, f"{paper_id}.json")

    # check if input file exists and output file doesn't
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} doesn't exist")
    if os.path.exists(output_file):
        print(f"{output_file} already exists!")

    # process PDF through Grobid -> TEI.XML
    client = GrobidClient(grobid_config)
    # TODO: compute PDF hash
    # TODO: add grobid version number to output
    client.process_pdf(input_file, temp_dir, "processFulltextDocument")

    # process TEI.XML -> JSON
    assert os.path.exists(tei_file)
    paper = convert_tei_xml_file_to_s2orc_json(tei_file)

    # write to file
    with open(output_file, "w") as outf:
        json.dump(paper.release_json(), outf, indent=4, sort_keys=False)

    return output_file
