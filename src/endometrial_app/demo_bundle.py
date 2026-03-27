from __future__ import annotations

import random
import tempfile
import zipfile
from pathlib import Path


DEMO_BUNDLE_ROUTE = "/downloads/demo-pack"
DEMO_BUNDLE_NAME = "endometrial-demo-test-images.zip"


def collect_demo_samples(project_root: Path) -> list[Path]:
    samples_dir = project_root / "assets" / "demo_samples"
    infected = sorted(samples_dir.glob("infected_*.jpg"))
    uninfected = sorted(samples_dir.glob("uninfected_*.jpg"))

    ordered_paths: list[Path] = []
    for infected_path, uninfected_path in zip(infected, uninfected):
        ordered_paths.extend([infected_path, uninfected_path])
    return ordered_paths


def demo_bundle_filename() -> str:
    return DEMO_BUNDLE_NAME


def build_demo_bundle_entries(project_root: Path) -> list[tuple[Path, str]]:
    sample_paths = collect_demo_samples(project_root)
    shuffled_paths = sample_paths[:]
    random.Random(2026).shuffle(shuffled_paths)

    bundle_entries: list[tuple[Path, str]] = []
    for index, sample_file in enumerate(shuffled_paths, start=1):
        bundle_entries.append((sample_file, f"scan_{index:02d}{sample_file.suffix.lower()}"))
    return bundle_entries


def build_demo_bundle(project_root: Path) -> str:
    bundle_entries = build_demo_bundle_entries(project_root)
    with tempfile.NamedTemporaryFile(
        prefix="endometrial-demo-samples-",
        suffix=".zip",
        delete=False,
    ) as temporary_file:
        bundle_path = Path(temporary_file.name)

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        manifest_lines = [
            "Endometrial Infection Classification App",
            "",
            "Bundle contents:",
            "- 20 demo images from the held-out test split",
            "- neutral scan filenames with no class labels in the archive",
            "- randomized file order for more natural blind testing",
            "",
            "Intended use:",
            "These images are provided so users can test the deployed application without sourcing their own scans.",
            "The archive is intentionally unlabeled so filename cues do not hint at the expected model output.",
        ]
        archive.writestr("README.txt", "\n".join(manifest_lines))
        for sample_file, neutral_name in bundle_entries:
            archive.write(sample_file, arcname=f"demo_samples/{neutral_name}")

    return str(bundle_path)
