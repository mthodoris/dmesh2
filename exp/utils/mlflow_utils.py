'''
Thin MLflow integration used by the mvrecon_3d / pcrecon_3d / pcrecon_2d training
scripts. Tracking defaults to a local sqlite db (./mlflow.db, with artifacts
under ./mlruns) - MLflow 3.x deprecated the plain file store, so sqlite is the
recommended local-only backend. Set the MLFLOW_TRACKING_URI env var to point
at a remote/server backend instead - mlflow picks that up automatically,
nothing else needs to change.
'''

import os
import numbers

import mlflow


def flatten_dict(d, parent_key="", sep="."):
    '''
    MLflow params/tags are flat key->value maps, but our yaml configs are
    nested. Flatten "args.init_preal.real_lr" style keys so the whole config
    is visible as run params.
    '''
    items = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, key, sep=sep))
        elif isinstance(v, (list, tuple)):
            items[key] = str(v)
        else:
            items[key] = v
    return items


def init_mlflow_run(experiment_name: str, run_name: str, settings: dict):
    '''
    Starts (and returns) an MLflow run, logging the full settings dict as
    params. Call mlflow.end_run() when the script finishes (successfully or
    not) to close it out.
    '''
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run = mlflow.start_run(run_name=run_name)

    flat = flatten_dict(settings)
    # mlflow caps param values at 500 chars and rejects non-primitive types;
    # stringify anything that isn't already a bool/int/float/str.
    params = {}
    for k, v in flat.items():
        if not isinstance(v, (str, numbers.Number, bool)) or v is None:
            v = str(v)
        params[k] = str(v)[:500]
    mlflow.log_params(params)

    return run


class MetricWriter:
    '''
    Drop-in wrapper around torch.utils.tensorboard.SummaryWriter that mirrors
    every add_scalar call into the currently active MLflow run. Everything
    else (add_image, log_dir, ...) is forwarded straight to the underlying
    SummaryWriter so existing call sites don't need to change.
    '''

    def __init__(self, tb_writer, use_mlflow: bool):
        self._tb = tb_writer
        self._use_mlflow = use_mlflow

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        self._tb.add_scalar(tag, scalar_value, global_step, *args, **kwargs)
        if self._use_mlflow and scalar_value is not None:
            try:
                value = float(scalar_value)
            except (TypeError, ValueError):
                return
            # mlflow metric keys may not contain some tensorboard-friendly chars;
            # keep it simple and just swap slashes for underscores.
            mlflow.log_metric(tag.replace("/", "_"), value, step=global_step)

    def __getattr__(self, name):
        return getattr(self._tb, name)
