import json
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import requests


INVESTIGATION_QUERIES = [
    ("BASS-II", "Burning and Suppression of Solids II microgravity"),
    ("SAFFIRE", "Spacecraft Fire Safety Demonstration Saffire microgravity"),
    ("ACME-BRE", "Burning Rate Emulator ACME microgravity combustion"),
    ("ACME-CFIG", "Cool Flames Investigation with Gases ACME microgravity"),
    ("ACME-EFIELD", "Electric-Field Effects Laminar Diffusion Flames microgravity"),
    ("FLEX", "Flame Extinguishment Experiment droplet microgravity cool flame"),
    ("SAME", "Smoke and Aerosol Measurement Experiment microgravity"),
    ("DAFT", "Dust and Aerosol Measurement Feasibility Test spacecraft"),
    ("SPICE-SLICE", "Smoke Point In Co-flow Experiment microgravity MSG"),
]


def fetch_psi_s3_manifest(output_path: Path, max_keys_per_prefix: int = 40) -> dict:
    s3 = boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(signature_version=UNSIGNED, connect_timeout=10, read_timeout=15),
    )
    bucket_name = "nasa-psi"
    manifest = {
        "bucket": f"s3://{bucket_name}",
        "region": "us-west-2",
        "top_level_prefixes": [],
        "sampled_objects": [],
        "ntrs_combustion_papers": [],
        "nasa_image_library_flames": [],
    }

    try:
        top_resp = s3.list_objects_v2(Bucket=bucket_name, Delimiter="/", MaxKeys=200)
        prefixes = [p["Prefix"] for p in top_resp.get("CommonPrefixes", [])]
        manifest["top_level_prefixes"] = prefixes

        for prefix in prefixes[:10]:
            obj_resp = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys_per_prefix,
            )
            for item in obj_resp.get("Contents", []):
                manifest["sampled_objects"].append(
                    {
                        "key": item["Key"],
                        "size_bytes": int(item["Size"]),
                        "last_modified": item["LastModified"].isoformat(),
                    }
                )
    except Exception as exc:
        manifest["s3_warning"] = str(exc)

    for inv_code, query_str in INVESTIGATION_QUERIES:
        try:
            ntrs_resp = requests.get(
                "https://ntrs.nasa.gov/api/citations/search",
                params={"q": query_str, "page[size]": 5},
                timeout=12,
            )
            if ntrs_resp.status_code == 200:
                results = ntrs_resp.json().get("results", [])
                for item in results[:4]:
                    manifest["ntrs_combustion_papers"].append(
                        {
                            "investigation": inv_code,
                            "ntrs_id": item.get("id"),
                            "title": item.get("title", ""),
                            "abstract": (item.get("abstract") or "")[:500],
                            "center": (item.get("center") or {}).get("code", "GRC"),
                            "published_date": item.get("distributionDate", ""),
                        }
                    )
        except Exception:
            continue

    try:
        img_resp = requests.get(
            "https://images-api.nasa.gov/search",
            params={"q": "microgravity flame combustion", "media_type": "image"},
            timeout=12,
        )
        if img_resp.status_code == 200:
            items = img_resp.json().get("collection", {}).get("items", [])
            for it in items[:15]:
                data_block = (it.get("data") or [{}])[0]
                links_block = (it.get("links") or [{}])[0]
                if links_block.get("href"):
                    manifest["nasa_image_library_flames"].append(
                        {
                            "nasa_id": data_block.get("nasa_id", ""),
                            "title": data_block.get("title", ""),
                            "description": (data_block.get("description") or "")[:350],
                            "thumb_url": links_block.get("href", ""),
                        }
                    )
    except Exception as exc:
        manifest["images_warning"] = str(exc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
