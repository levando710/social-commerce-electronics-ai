"""Version an Electronics draft to correct an inherited phone/accessory description, without changing data."""
import argparse
import json
from pathlib import Path
import shutil

from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, sha256

INHERITED_DESCRIPTION = "Fixed catalog from the simulated phone/accessory benchmark; not a representative sample of all Amazon."
ELECTRONICS_DESCRIPTION = "Fixed Electronics catalog from the released SWAP/UNIBA benchmark; not a representative sample of all Amazon Electronics."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / 'data/external/benchmarks/connected_electronics_v1')
    parser.add_argument("--output", type=Path, default=ROOT / 'data/external/benchmarks/connected_electronics_v2')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Retain frozen versions")
    data = InteractionData(args.source)
    summary = json.loads((args.source / "summary.json").read_text(encoding="utf-8"))
    if summary["domain"] != "Electronics" or INHERITED_DESCRIPTION not in summary["limitations"]:
        raise ValueError("Not the expected Electronics draft")
    args.output.mkdir(parents=True)
    for name in data.manifest["artifacts_sha256"]:
        if name != "summary.json":
            shutil.copyfile(args.source / name, args.output / name)
            if sha256(args.source / name) != sha256(args.output / name): raise ValueError("Dataset bytes changed")
    summary["limitations"] = [ELECTRONICS_DESCRIPTION if x == INHERITED_DESCRIPTION else x for x in summary["limitations"]]
    summary["annotation_correction"] = "Only the inherited catalog-domain description changed; interactions, labels, IDs and metadata bytes are unchanged from v1."
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {**data.manifest, "parent_manifest_sha256": sha256(args.source / "manifest.json"),
        "source_sha256": {**data.manifest["source_sha256"], "annotation_corrector": sha256(Path(__file__))},
        "artifacts_sha256": {name: sha256(args.output / name) for name in data.manifest["artifacts_sha256"]}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    final = InteractionData(args.output)
    print(json.dumps({"users": final.n_users, "items": final.n_items, "data_unchanged": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
