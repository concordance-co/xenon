from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from research.synthetic_market.synthetic_market_patching_runner import _parse_component_indices_spec
from research.synthetic_market.synthetic_market_behavior_runner import SyntheticMarketBehaviorConfig


DEFAULT_LAMBDA_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
DEFAULT_RANDOM_CONTROL_SEEDS = (11, 17, 23, 29, 31)


@dataclass(slots=True)
class SyntheticMarketBehaviorPlanItem:
    run_name: str
    sweep_kind: str
    sweep_value: str
    description: str
    config: SyntheticMarketBehaviorConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "sweep_kind": self.sweep_kind,
            "sweep_value": self.sweep_value,
            "description": self.description,
            "patch_mode": self.config.patch_mode,
            "pair_mode": self.config.pair_mode,
            "strength": float(self.config.strength),
            "random_seed": int(self.config.random_seed),
            "target_layers": [int(layer) for layer in self.config.target_layers],
            "component_indices_by_layer": {
                str(layer): [int(index) for index in indices]
                for layer, indices in self.config.component_indices_by_layer.items()
            },
            "components_per_layer": int(self.config.components_per_layer),
            "phase_name": self.config.phase_name,
            "context_variant": self.config.context_variant,
            "selection_strategy": self.config.selection_strategy,
            "limit": self.config.limit,
            "family_allowlist": list(self.config.family_allowlist),
            "pair_metric": self.config.pair_metric,
            "min_pair_gap": float(self.config.min_pair_gap),
            "generate_source_behavior": bool(self.config.generate_source_behavior),
            "batch_size": int(self.config.batch_size),
            "direction_name": self.config.direction_name,
            "max_tokens": int(self.config.max_tokens),
            "temperature": float(self.config.temperature),
            "top_p": float(self.config.top_p),
            "top_k": int(self.config.top_k),
            "tool_schema_mode": self.config.tool_schema_mode,
            "tool_choice": self.config.tool_choice,
            "add_generation_prompt": bool(self.config.add_generation_prompt),
            "enforce_eager": bool(self.config.enforce_eager),
            "enable_chunked_prefill": bool(self.config.enable_chunked_prefill),
        }


def _slugify_float(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def _component_map_from_config(config: SyntheticMarketBehaviorConfig) -> dict[int, tuple[int, ...]]:
    if config.component_indices_by_layer:
        return {
            int(layer): tuple(int(index) for index in indices)
            for layer, indices in sorted(config.component_indices_by_layer.items())
        }
    default_indices = tuple(range(max(1, int(config.components_per_layer))))
    return {int(layer): default_indices for layer in config.target_layers}


def _shift_component_map(
    component_map: dict[int, tuple[int, ...]],
    *,
    offset: int,
) -> dict[int, tuple[int, ...]] | None:
    shifted: dict[int, tuple[int, ...]] = {}
    for layer, indices in component_map.items():
        new_indices = tuple(int(index) + int(offset) for index in indices)
        if any(index < 0 for index in new_indices):
            return None
        shifted[int(layer)] = new_indices
    return shifted


def _resize_component_map(
    component_map: dict[int, tuple[int, ...]],
    *,
    size: int,
) -> dict[int, tuple[int, ...]] | None:
    target_size = int(size)
    if target_size <= 0:
        return None
    resized: dict[int, tuple[int, ...]] = {}
    for layer, indices in component_map.items():
        if not indices:
            return None
        resized[int(layer)] = tuple(int(index) for index in indices[:target_size])
    return resized


def _dedupe_component_variants(
    variants: list[tuple[str, dict[int, tuple[int, ...]], str]],
) -> list[tuple[str, dict[int, tuple[int, ...]], str]]:
    deduped: list[tuple[str, dict[int, tuple[int, ...]], str]] = []
    seen: set[tuple[tuple[int, tuple[int, ...]], ...]] = set()
    for suffix, component_map, description in variants:
        key = tuple((int(layer), tuple(int(index) for index in indices)) for layer, indices in sorted(component_map.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((suffix, component_map, description))
    return deduped


def build_behavior_robustness_battery(
    base_config: SyntheticMarketBehaviorConfig,
    *,
    run_name_prefix: str,
    lambda_sweep: tuple[float, ...] = DEFAULT_LAMBDA_SWEEP,
    neighboring_component_offsets: tuple[int, ...] = (0,),
    subspace_sizes: tuple[int, ...] = (),
    random_control_seeds: tuple[int, ...] = DEFAULT_RANDOM_CONTROL_SEEDS,
    pair_modes: tuple[str, ...] = (),
) -> list[SyntheticMarketBehaviorPlanItem]:
    if not base_config.patch_enabled:
        raise ValueError("Base behavior config must have patching enabled to build a robustness battery")
    prefix = str(run_name_prefix).strip()
    if not prefix:
        raise ValueError("run_name_prefix must be non-empty")

    base_component_map = _component_map_from_config(base_config)
    component_variants: list[tuple[str, dict[int, tuple[int, ...]], str]] = [("base", base_component_map, "target components")]
    for offset in neighboring_component_offsets:
        shifted = _shift_component_map(base_component_map, offset=int(offset))
        if shifted is None:
            continue
        sign = "p" if int(offset) >= 0 else "m"
        component_variants.append(
            (
                f"shift_{sign}{abs(int(offset))}",
                shifted,
                f"neighbor component offset {int(offset)}",
            )
        )
    for size in subspace_sizes:
        resized = _resize_component_map(base_component_map, size=int(size))
        if resized is None:
            continue
        component_variants.append((f"k{int(size)}", resized, f"subspace size {int(size)}"))
    component_variants = _dedupe_component_variants(component_variants)

    effective_pair_modes = pair_modes or ((base_config.pair_mode,) if base_config.pair_mode else ("",))
    strengths = tuple(float(value) for value in lambda_sweep)
    plan: list[SyntheticMarketBehaviorPlanItem] = []

    def _build_config(
        *,
        patch_mode: str,
        strength: float,
        random_seed: int,
        pair_mode: str,
        component_map: dict[int, tuple[int, ...]],
    ) -> SyntheticMarketBehaviorConfig:
        max_components = max((len(indices) for indices in component_map.values()), default=0)
        return replace(
            base_config,
            patch_mode=str(patch_mode),
            strength=float(strength),
            random_seed=int(random_seed),
            pair_mode=str(pair_mode),
            component_indices_by_layer=component_map,
            components_per_layer=max_components if max_components > 0 else int(base_config.components_per_layer),
        )

    for pair_mode in effective_pair_modes:
        pair_mode_suffix = str(pair_mode).strip().lower() or "unpaired"
        for component_suffix, component_map, component_description in component_variants:
            for strength in strengths:
                strength_suffix = f"lam_{_slugify_float(strength)}"
                run_name = (
                    f"{prefix}_{base_config.patch_mode}_{pair_mode_suffix}_{component_suffix}_{strength_suffix}"
                )
                config = _build_config(
                    patch_mode=base_config.patch_mode,
                    strength=float(strength),
                    random_seed=int(base_config.random_seed),
                    pair_mode=str(pair_mode),
                    component_map=component_map,
                )
                plan.append(
                    SyntheticMarketBehaviorPlanItem(
                        run_name=run_name,
                        sweep_kind="targeted",
                        sweep_value=f"{component_suffix}:{strength_suffix}",
                        description=f"{component_description}; lambda={float(strength):g}",
                        config=config,
                    )
                )

            for random_seed in random_control_seeds:
                for strength in strengths:
                    if float(strength) == 0.0:
                        continue
                    run_name = (
                        f"{prefix}_random_control_{pair_mode_suffix}_{component_suffix}_"
                        f"seed_{int(random_seed)}_lam_{_slugify_float(strength)}"
                    )
                    config = _build_config(
                        patch_mode="random_control",
                        strength=float(strength),
                        random_seed=int(random_seed),
                        pair_mode=str(pair_mode),
                        component_map=component_map,
                    )
                    plan.append(
                        SyntheticMarketBehaviorPlanItem(
                            run_name=run_name,
                            sweep_kind="random_control",
                            sweep_value=f"{component_suffix}:seed{int(random_seed)}:lam{float(strength):g}",
                            description=(
                                f"matched random control; {component_description}; "
                                f"seed={int(random_seed)}; lambda={float(strength):g}"
                            ),
                            config=config,
                        )
                    )

    return plan


def build_behavior_baseline_plan(
    base_config: SyntheticMarketBehaviorConfig,
    *,
    run_name_prefix: str,
    pair_modes: tuple[str, ...] = (),
) -> list[SyntheticMarketBehaviorPlanItem]:
    prefix = str(run_name_prefix).strip()
    if not prefix:
        raise ValueError("run_name_prefix must be non-empty")
    effective_pair_modes = pair_modes or ((base_config.pair_mode,) if base_config.pair_mode else ("",))
    plan: list[SyntheticMarketBehaviorPlanItem] = []
    for pair_mode in effective_pair_modes:
        pair_mode_suffix = str(pair_mode).strip().lower() or "unpaired"
        run_name = f"{prefix}_baseline_{pair_mode_suffix}"
        config = replace(
            base_config,
            patch_mode="none",
            pair_mode=str(pair_mode),
            strength=0.0,
        )
        plan.append(
            SyntheticMarketBehaviorPlanItem(
                run_name=run_name,
                sweep_kind="baseline",
                sweep_value=pair_mode_suffix,
                description=f"baseline no-patch run for pair_mode={pair_mode_suffix}",
                config=config,
            )
        )
    return plan


def build_behavior_robustness_matrix(
    base_config: SyntheticMarketBehaviorConfig,
    *,
    run_name_prefix: str,
    lambda_sweep: tuple[float, ...] = DEFAULT_LAMBDA_SWEEP,
    neighboring_component_offsets: tuple[int, ...] = (0,),
    subspace_sizes: tuple[int, ...] = (),
    random_control_seeds: tuple[int, ...] = DEFAULT_RANDOM_CONTROL_SEEDS,
    pair_modes: tuple[str, ...] = (),
    include_baselines: bool = True,
) -> list[SyntheticMarketBehaviorPlanItem]:
    plan: list[SyntheticMarketBehaviorPlanItem] = []
    if include_baselines:
        plan.extend(
            build_behavior_baseline_plan(
                base_config,
                run_name_prefix=run_name_prefix,
                pair_modes=pair_modes,
            )
        )
    plan.extend(
        build_behavior_robustness_battery(
            base_config,
            run_name_prefix=run_name_prefix,
            lambda_sweep=lambda_sweep,
            neighboring_component_offsets=neighboring_component_offsets,
            subspace_sizes=subspace_sizes,
            random_control_seeds=random_control_seeds,
            pair_modes=pair_modes,
        )
    )
    return plan


def build_behavior_robustness_payload(
    base_config: SyntheticMarketBehaviorConfig,
    *,
    run_name_prefix: str,
    lambda_sweep: tuple[float, ...] = DEFAULT_LAMBDA_SWEEP,
    neighboring_component_offsets: tuple[int, ...] = (0,),
    subspace_sizes: tuple[int, ...] = (),
    random_control_seeds: tuple[int, ...] = DEFAULT_RANDOM_CONTROL_SEEDS,
    pair_modes: tuple[str, ...] = (),
    include_baselines: bool = True,
) -> dict[str, Any]:
    plan = build_behavior_robustness_matrix(
        base_config,
        run_name_prefix=run_name_prefix,
        lambda_sweep=lambda_sweep,
        neighboring_component_offsets=neighboring_component_offsets,
        subspace_sizes=subspace_sizes,
        random_control_seeds=random_control_seeds,
        pair_modes=pair_modes,
        include_baselines=include_baselines,
    )
    return {
        "run_name_prefix": str(run_name_prefix),
        "count": len(plan),
        "counts_by_sweep_kind": {
            sweep_kind: sum(1 for item in plan if item.sweep_kind == sweep_kind)
            for sweep_kind in sorted({item.sweep_kind for item in plan})
        },
        "runs": [item.to_dict() for item in plan],
    }


def _parse_float_csv(text: str) -> tuple[float, ...]:
    tokens = [token.strip() for token in str(text).split(",") if token.strip()]
    return tuple(float(token) for token in tokens)


def _parse_int_csv(text: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in str(text).split(",") if token.strip()]
    return tuple(int(token) for token in tokens)


def _parse_str_csv(text: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in str(text).split(",") if token.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a reproducible synthetic-market patching robustness battery.")
    parser.add_argument("--run-name-prefix", required=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional JSON path to write the expanded robustness plan.",
    )
    parser.add_argument("--phase-name", default="phase15_market_basis_discovery_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis_results/synthetic_market_behavior"),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--context-variant", default="market_only")
    parser.add_argument("--order-mode", default="selection_rank_asc")
    parser.add_argument("--selection-strategy", default="ordered")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--family-allowlist", default="")
    parser.add_argument("--pair-metric", default="")
    parser.add_argument("--pair-mode", default="")
    parser.add_argument("--pair-modes", default="")
    parser.add_argument("--min-pair-gap", type=float, default=0.0)
    parser.add_argument("--generate-source-behavior", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-mode", default="project_out")
    parser.add_argument("--target-layers", default="4")
    parser.add_argument("--components-per-layer", type=int, default=4)
    parser.add_argument("--component-indices", default="")
    parser.add_argument("--direction-name", default="")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--tool-schema-mode", default="")
    parser.add_argument("--tool-choice", default="")
    parser.add_argument("--add-generation-prompt", action="store_true")
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--lambda-sweep", default="0.0,0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--neighboring-component-offsets", default="0")
    parser.add_argument("--subspace-sizes", default="")
    parser.add_argument("--random-control-seeds", default="11,17,23,29,31")
    parser.add_argument("--no-include-baselines", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    target_layers = tuple(int(token) for token in args.target_layers.split(",") if token.strip())
    component_indices_by_layer = _parse_component_indices_spec(
        args.component_indices,
        target_layers=target_layers or (4,),
    )
    family_allowlist = _parse_str_csv(args.family_allowlist)
    pair_modes = _parse_str_csv(args.pair_modes)
    if not pair_modes and args.pair_mode.strip():
        pair_modes = (args.pair_mode.strip(),)
    base_config = SyntheticMarketBehaviorConfig(
        phase_name=args.phase_name,
        output_dir=args.output_dir,
        model_id=args.model_id,
        context_variant=args.context_variant,
        order_mode=args.order_mode,
        selection_strategy=args.selection_strategy,
        limit=args.limit if args.limit > 0 else None,
        family_allowlist=family_allowlist,
        pair_metric=args.pair_metric,
        pair_mode=args.pair_mode,
        min_pair_gap=float(args.min_pair_gap),
        generate_source_behavior=bool(args.generate_source_behavior),
        batch_size=max(1, int(args.batch_size)),
        patch_mode=args.patch_mode,
        target_layers=target_layers or (4,),
        components_per_layer=int(args.components_per_layer),
        component_indices_by_layer=component_indices_by_layer,
        direction_name=args.direction_name,
        strength=float(args.strength),
        random_seed=int(args.random_seed),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        tool_schema_mode=args.tool_schema_mode,
        tool_choice=args.tool_choice,
        add_generation_prompt=bool(args.add_generation_prompt),
        enforce_eager=not bool(args.no_enforce_eager),
        enable_chunked_prefill=bool(args.enable_chunked_prefill),
    )
    payload = build_behavior_robustness_payload(
        base_config,
        run_name_prefix=args.run_name_prefix,
        lambda_sweep=_parse_float_csv(args.lambda_sweep),
        neighboring_component_offsets=_parse_int_csv(args.neighboring_component_offsets),
        subspace_sizes=_parse_int_csv(args.subspace_sizes),
        random_control_seeds=_parse_int_csv(args.random_control_seeds),
        pair_modes=pair_modes,
        include_baselines=not bool(args.no_include_baselines),
    )
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
