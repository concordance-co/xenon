from projects.DX_TERMINAL.synthetic_market.shared.modal_capture import (
    resolve_synthetic_capture_output_subdir,
    resolve_synthetic_capture_relation,
)


def test_resolve_synthetic_capture_relation_defaults_to_capture_view() -> None:
    assert (
        resolve_synthetic_capture_relation("phase 22 smoke")
        == "synthetic_market_phase_22_smoke_capture_v0"
    )


def test_resolve_synthetic_capture_relation_respects_explicit_relation() -> None:
    assert (
        resolve_synthetic_capture_relation(
            "phase 22 smoke",
            "custom_synthetic_capture_relation",
        )
        == "custom_synthetic_capture_relation"
    )


def test_resolve_synthetic_capture_output_subdir_defaults_to_project_path() -> None:
    assert (
        resolve_synthetic_capture_output_subdir("phase22_smoke_v1")
        == "projects/DX_TERMINAL/synthetic_market/captures/phase22_smoke_v1"
    )


def test_resolve_synthetic_capture_output_subdir_respects_explicit_value() -> None:
    assert (
        resolve_synthetic_capture_output_subdir(
            "phase22_smoke_v1",
            "custom/output/subdir",
        )
        == "custom/output/subdir"
    )
