"""Offline regression for import-dependent bundled distribution discovery."""
import importlib.metadata
import importlib.metadata._adapters
from pathlib import Path
import sys

import pytest


@pytest.fixture
def inventory(monkeypatch):
    import pipelines_v2.core.provenance as module
    monkeypatch.setattr(module, "_PACKAGE_CACHE", None)
    module._owned_distribution_metadata.cache_clear()
    return module


def write_distribution(root, name, version, *, owned=()):
    path = root / f"{name}.dist-info"
    path.mkdir(parents=True, exist_ok=True)
    (path / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    (path / "RECORD").write_text("\n".join(f"{name}.dist-info/{field},," for field in ("METADATA", "RECORD")) + "\n" + "\n".join(f"{item},," for item in owned))
    return path


def test_vendor_import_and_duplicate_search_path_do_not_change_inventory(tmp_path, monkeypatch, inventory):
    vendor = tmp_path / "owner" / "_vendor"
    write_distribution(vendor, "bundled", "1.0")
    write_distribution(tmp_path, "owner", "2.0", owned=("owner/_vendor/bundled.dist-info/METADATA",))
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    before = inventory._installed_packages()
    assert before == (("bundled", "1.0"), ("owner", "2.0"))
    monkeypatch.setattr(sys, "path", [str(tmp_path), str(vendor), str(tmp_path)])
    assert inventory._installed_packages() == before


@pytest.mark.parametrize("target", ["outer", "vendor", "new_install", "new_vendor"])
def test_true_package_changes_invalidate(tmp_path, monkeypatch, inventory, target):
    vendor = tmp_path / "owner" / "_vendor"
    write_distribution(vendor, "bundled", "1.0")
    owned = ("owner/_vendor/bundled.dist-info/METADATA",)
    write_distribution(tmp_path, "owner", "2.0", owned=owned)
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    before = inventory._installed_packages()
    if target == "outer":
        write_distribution(tmp_path, "owner", "3.0", owned=owned)
    elif target == "vendor":
        write_distribution(vendor, "bundled", "2.0")
    elif target == "new_install":
        write_distribution(tmp_path, "new-package", "1.0")
    else:
        write_distribution(vendor, "new-vendor", "1.0")
        write_distribution(tmp_path, "owner", "2.0", owned=(*owned, "owner/_vendor/new-vendor.dist-info/METADATA"))
    assert inventory._installed_packages() != before


def test_distinct_providers_retained_and_record_scan_cached(tmp_path, monkeypatch, inventory):
    other = tmp_path / "other"
    write_distribution(tmp_path, "same-package", "1.0")
    write_distribution(other, "same-package", "1.0")
    monkeypatch.setattr(sys, "path", [str(tmp_path), str(other)])
    assert inventory._installed_packages() == (("same-package", "1.0"), ("same-package", "1.0"))
    misses = inventory._owned_distribution_metadata.cache_info().misses
    inventory._installed_packages()
    assert inventory._owned_distribution_metadata.cache_info().misses == misses
    write_distribution(other, "same-package", "2.0")
    assert inventory._installed_packages() == (("same-package", "1.0"), ("same-package", "2.0"))


def test_capture_shards_merge_across_vendor_import(tmp_path, monkeypatch, inventory):
    from pipelines_v2.api import CaptureSpec, Dataset, Example, LocalArtifactStore, ResidualSite, ToyEngine
    from pipelines_v2.runtime.remote_executor import execute_remote, merge_remote_shards
    import pipelines_v2.core.provenance as provenance
    packages = tmp_path / "site-packages"
    vendor = packages / "owner" / "_vendor"
    write_distribution(vendor, "bundled", "1.0")
    write_distribution(packages, "owner", "2.0", owned=("owner/_vendor/bundled.dist-info/METADATA",))
    paths = [str(packages)]
    discover = importlib.metadata.distributions
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: discover(path=paths))
    operation = CaptureSpec(engine=ToyEngine(), dataset=Dataset.from_examples([Example(key="a", prompt="one row")]),
                            sites=[ResidualSite(name="last", site="resid_post", layers=[0])])
    store = LocalArtifactStore(tmp_path / "artifacts")
    args = dict(runner_config={"kind": "test"}, store_config=store.identity(), spec_payload=operation.to_dict())
    shards = []
    for index in range(2):
        shards.append(execute_remote(**args, workflow_context={"execution_shard": {"index": index, "count": 2}}))
        paths.append(str(vendor))  # the real setuptools import exposes this path
    assert shards[0]["metadata"]["parent_provenance_v1"] == shards[1]["metadata"]["parent_provenance_v1"]
    merged = merge_remote_shards(**args, shard_manifests=shards)
    assert merged["metadata"][provenance.KEY]["runtime"] == shards[0]["metadata"][provenance.KEY]["runtime"]
