import mlflow

from privacy_and_grokking.utils import Logger, setup_mlflow
from privacy_and_grokking.visualize.handler import visualization_multi, visualization_single
from privacy_and_grokking.visualize.visualizations import MULTI_VIZ_NAMES, SINGLE_VIZ_NAMES


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
):
    setup_mlflow(exp_name)

    visualizations = extract_visualizations(MULTI_VIZ_NAMES, include=include, exclude=exclude)

    if run_ids is None and tag is None:
        raise ValueError("Either run_ids or tag must be provided.")
    if run_ids is not None and tag is not None:
        raise ValueError("Only one of run_ids or tag should be provided, not both.")

    if tag is not None:
        # e.g. tag: "visualize='true'"
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[client.get_experiment_by_name(exp_name).experiment_id],
            filter_string=f"tags.{tag}",
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        )
        run_ids = [run.info.run_id for run in runs]

    if not run_ids:
        raise ValueError("No runs found for visualization.")

    with Logger() as logger:
        logger.info(
            "Starting multi-run visualization.",
            run_ids=run_ids,
            visualizations=sorted(visualizations),
        )
        visualization_multi(run_ids, visualizations)
        logger.info("Multi-run visualization complete.", run_ids=run_ids)
