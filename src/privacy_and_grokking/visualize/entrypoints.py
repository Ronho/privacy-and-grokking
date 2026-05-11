import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow

from privacy_and_grokking.utils import Logger, setup_mlflow
from privacy_and_grokking.visualize.handler import (
    visualization_multi,
    visualization_multi_groups,
    visualization_single,
)
from privacy_and_grokking.visualize.visualizations import MULTI_VIZ_NAMES, SINGLE_VIZ_NAMES

_MAX_CONCURRENT_GROUP_PROCESSES = 20


def extract_visualizations(
    viz_names: list[str], include: list[str] | None = None, exclude: list[str] | None = None
):
    if include is not None:
        for name in include:
            if name not in viz_names:
                raise ValueError(
                    f"Visualization '{name}' is not a valid visualization name. Valid options are: {', '.join(viz_names)}"
                )
        effective_include = include
    else:
        effective_include = viz_names

    if exclude is not None:
        for name in exclude:
            if name not in viz_names:
                raise ValueError(
                    f"Visualization '{name}' is not a valid visualization name. Valid options are: {', '.join(viz_names)}"
                )
        effective_include = [n for n in effective_include if n not in exclude]
    return effective_include


def visualization_single_handler(
    exp_name: str,
    run_id: str,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
):
    setup_mlflow(exp_name)

    visualizations = extract_visualizations(SINGLE_VIZ_NAMES, include=include, exclude=exclude)

    with Logger() as logger:
        logger.info(
            "Starting visualization.",
            run_id=run_id,
            visualizations=sorted(visualizations),
        )
        visualization_single(run_id, visualizations)
        logger.info("Visualization complete.", run_id=run_id)


def visualization_multi_handler(
    exp_name: str,
    run_ids: list[str] | None = None,
    tag: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    postfix: str | None = None,
    group: bool = False,
    aggregate: bool = False,
):
    logger = Logger.get()
    setup_mlflow(exp_name)

    if run_ids is None and tag is None:
        raise ValueError("Either run_ids or tag must be provided.")
    if run_ids is not None and tag is not None:
        raise ValueError("Only one of run_ids or tag should be provided, not both.")

    if tag is not None:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[client.get_experiment_by_name(exp_name).experiment_id],
            filter_string=f"tags.{tag}",
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        )
        resolved_run_ids = [run.info.run_id for run in runs]
    else:
        resolved_run_ids = list(run_ids)

    if not resolved_run_ids:
        raise ValueError("No runs found for visualization.")

    if aggregate and not group:
        raise ValueError("--aggregate requires --group to be set as well.")

    if group:
        client = mlflow.tracking.MlflowClient()
        groups: dict[str, list[str]] = {}
        for rid in resolved_run_ids:
            run = client.get_run(rid)
            name = run.data.tags.get("mlflow.runName", rid)
            groups.setdefault(name, []).append(rid)

        logger.info(
            f"Grouping {len(resolved_run_ids)} runs into {len(groups)} groups.",
            extra={"groups": {k: len(v) for k, v in groups.items()}},
        )

        if aggregate:
            visualizations = extract_visualizations(
                MULTI_VIZ_NAMES, include=include, exclude=exclude
            )
            logger.info(
                "Producing aggregated group figure.",
                visualizations=sorted(visualizations),
            )
            visualization_multi_groups(groups, visualizations, postfix=postfix)
            logger.info("Aggregate group visualization complete.")
            return

        def _run_group(group_name: str, group_run_ids: list[str]) -> str:
            cmd = [
                sys.executable,
                "-m",
                "privacy_and_grokking.cli",
                "visualize-multi",
                exp_name,
            ]
            for rid in group_run_ids:
                cmd += ["--run-ids", rid]
            if include:
                for v in include:
                    cmd += ["--include", v]
            if exclude:
                for v in exclude:
                    cmd += ["--exclude", v]
            if postfix:
                cmd += ["--postfix", postfix]
            subprocess.run(cmd, check=True)
            return group_name

        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_GROUP_PROCESSES) as executor:
            futures = {
                executor.submit(_run_group, group_name, group_run_ids): group_name
                for group_name, group_run_ids in groups.items()
            }
            for future in as_completed(futures):
                group_name = futures[future]
                try:
                    future.result()
                    logger.info(f"Group '{group_name}' visualization completed.")
                except subprocess.CalledProcessError as exc:
                    logger.error(
                        f"Group '{group_name}' visualization failed.",
                        extra={"returncode": exc.returncode},
                    )

        logger.info("All group visualization processes completed.")
        return

    visualizations = extract_visualizations(MULTI_VIZ_NAMES, include=include, exclude=exclude)
    logger.info(
        "Starting multi-run visualization.",
        run_ids=resolved_run_ids,
        visualizations=sorted(visualizations),
    )
    visualization_multi(resolved_run_ids, visualizations, postfix=postfix)
    logger.info("Multi-run visualization complete.", run_ids=resolved_run_ids)
