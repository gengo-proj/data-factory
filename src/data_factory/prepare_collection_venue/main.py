import os
import re
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path
from typing import cast

import sienna
import yaml

from data_factory.utils import paper_url_to_uuid

strmo2intmo = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def collection_id_to_venue_id(collection_id: str) -> str:
    # removes the year part
    if re.search("[A-Za-z]+[0-9]?[A-Za-z]+[0-9]*", collection_id):
        return re.search("[A-Za-z]+[0-9]?[A-Za-z]+[0-9]*", collection_id).group(0)
    else:
        return re.search("^[A-Za-z]?", collection_id).group(0)


def main(yaml_dir: str, xml_dir: str, output_dir: str, enriched_papers_dir: str | None):
    yaml_path = Path(yaml_dir)
    xml_path = Path(xml_dir)
    output_path = Path(output_dir)

    collection_id_filter = (
        os.listdir(enriched_papers_dir) if enriched_papers_dir is not None else None
    )

    (output_path / "collections").mkdir(exist_ok=True)
    (output_path / "volumes").mkdir(exist_ok=True)

    venue2info: dict[str, dict[str, str | bool]] = {}
    for venue_path in yaml_path.glob("venues/*.yaml"):
        venue_name = venue_path.stem
        with open(venue_path) as f:
            venue_info = yaml.load(f, Loader=yaml.CLoader)
        venue2info[venue_name] = venue_info
        if "oldstyle_letter" in venue_info.keys():
            venue2info[venue_info["oldstyle_letter"]] = venue_info

    for collection_file_path in xml_path.glob("*.xml"):
        print(collection_file_path)
        # build collection data
        tree = ET.parse(collection_file_path)
        root = tree.getroot()

        collection = {}
        collection_id = root.attrib["id"]
        collection["collection_id"] = collection_id
        collection_name = collection_id_to_venue_id(collection_id)

        if collection_id_filter is not None:
            if collection_id not in collection_id_filter:
                print("Skip")
                continue
            else:
                print("Use")

        collection_name = "figlang" if collection_name == "flp" else collection_name
        collection_name = (
            "nlp4pi" if collection_name == "nlp4posimpact" else collection_name
        )

        collection["collection_name"] = collection_name
        collection["collection_long_name"] = venue2info[collection_name]["name"]
        collection["year"] = (
            int(root.find(".//year").text)
            if hasattr(root.find(".//year"), "text")
            else collection_id[:4]
        )

        collection["is_acl"] = venue2info[collection_name].get("is_acl", None)
        collection["is_toplevel"] = venue2info[collection_name].get("is_toplevel", None)
        collection["acronym"] = venue2info[collection_name]["acronym"]
        collection["oldstyle_letter"] = venue2info[collection_name].get(
            "oldstyle_letter", None
        )
        collection["volume_ids"] = [
            f"{collection_id}-{_v.attrib['id']}" for _v in root.findall("volume")
        ]
        if root.find("event") is not None:
            collection["event_id"] = root.find("event").attrib["id"]
            if root.find("event").find("colocated") is not None:
                collection["colocated_volume_ids"] = [
                    _v.text
                    for _v in root.find("event").find("colocated").findall("volume-id")
                ]
        else:
            collection["event_id"] = None
            collection["colocated_volume_ids"] = None

        sienna.save(
            collection,
            str(output_path / "collections" / f"{collection_id}.json"),
            indent=2,
        )

        # build volume data
        volume = {}
        for _volume in root.findall("volume"):
            volume_id = f"{collection_id}-{_volume.attrib['id']}"
            volume["volume_id"] = volume_id
            volume["booktitle"] = _volume.find("meta").find("booktitle").text
            volume["editors"] = [
                {"first": editor.find("first").text, "last": editor.find("last").text}
                for editor in _volume.find("meta").findall("editor")
            ]
            _meta = _volume.find("meta")
            volume["address"] = (
                _meta.find("address").text if _meta.find("address") else None
            )
            if _meta.find("month"):
                volume["month"] = (
                    int(_meta.find("month").text)
                    if _meta.find("month").text.isdigit()
                    else strmo2intmo[_meta.find("month").text.lower()]
                )
            else:
                volume["month"] = None
            volume["year"] = int(_meta.find("year").text)
            if hasattr(_meta.find("url"), "text"):
                volume["url"] = _meta.find("url").text
            else:
                volume["url"] = volume_id
            volume["venue"] = _meta.find("venue").text
            volume["frontmatter"] = (
                {
                    "url": _meta.find("frontmatter").find("url").text,
                    "bibkey": _meta.find("bibkey").find("bibkey").text,
                }
                if _meta.find("frontmatter")
                else None
            )

            volume["paper_ids"] = []
            for _paper in _volume.findall("paper"):
                paper_id = int(_paper.attrib["id"])
                if hasattr(_paper.find("url"), "text"):
                    paper_url = cast(str, _paper.find("url").text)
                else:
                    paper_url = f"{volume_id}{str(paper_id).rjust(3, '0')}"
                volume["paper_ids"].append(str(paper_url_to_uuid(paper_url)))

            sienna.save(
                volume, str(output_path / "volumes" / f"{volume_id}.json"), indent=2
            )

    # find duplicates: both new collection id and old style letter files exist
    for collection_path in (output_path / "collections").glob("[0-9]*.json"):
        collection = sienna.load(str(collection_path))
        if collection["oldstyle_letter"] and (
            collection["oldstyle_letter"] != collection["collection_id"]
        ):
            # load old file if exists
            old_file_path = (
                output_path
                / "collections"
                / f"{collection['oldstyle_letter']}{str(collection['year'])[-2:]}.json"
            )
            if old_file_path.exists():
                print("Duplicate found")
                print(f"Old: {old_file_path}")
                print(f"New: {collection_path}")
                old_collection = sienna.load(str(old_file_path))

                # update volume ids
                assert len(collection["volume_ids"]) == 0
                assert len(old_collection["volume_ids"]) != 0
                collection["volume_ids"] = old_collection["volume_ids"]

                # update year
                collection["year"] = old_collection["year"]

                # save new one
                sienna.save(collection, str(collection_path), indent=2)

                # remove old one
                old_file_path.unlink()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--yaml-dir", type=str, required=True)
    parser.add_argument("--xml-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--enriched-papers-dir", type=str, required=False, default=None)
    args = parser.parse_args()

    main(args.yaml_dir, args.xml_dir, args.output_dir, args.enriched_papers_dir)
