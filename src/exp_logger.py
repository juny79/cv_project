# src/exp_logger.py  (W&B + MLflow 방탄 트래커)
from typing import Dict, Any
import os

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

def _flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        nk = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, nk, sep))
        else:
            out[nk] = v
    return out

class Tracker:
    def __init__(self, cfg: Dict[str, Any], fold_id: int):
        self.cfg = cfg
        self.fold_id = fold_id
        trk = cfg.get("tracking", {})
        wb = trk.get("wandb", {})
        mf = trk.get("mlflow", {})
        self.wb_enabled = bool(wb.get("enabled", False) and _HAS_WANDB)
        self.mf_enabled = bool(mf.get("enabled", False) and _HAS_MLFLOW)
        self.wb_run = None
        self.mf_active = False
        self._param_logged = False
        self._gstep = 0

        # MLflow
        if self.mf_enabled:
            try:
                mlflow.set_tracking_uri(mf.get("tracking_uri", "file:./mlruns"))
                mlflow.set_experiment(mf.get("experiment", "default"))
                mlflow.start_run(run_name=f"{mf.get('run_name_prefix','train')}-fold{fold_id}")
                self.mf_active = True
            except Exception as e:
                print(f"[WARN] mlflow init failed: {e}")
                self.mf_enabled = False
                self.mf_active = False

        # W&B
        if self.wb_enabled:
            try:
                settings = wandb.Settings(start_method="thread")
                self.wb_run = wandb.init(
                    project=wb.get("project", "doccls"),
                    entity=wb.get("entity", None),
                    name=f"{wb.get('run_name_prefix','train')}-fold{fold_id}",
                    tags=wb.get("tags", []),
                    config=cfg,
                    settings=settings
                )
            except Exception as e1:
                try:
                    print(f"[WARN] wandb.init failed with entity={wb.get('entity')}: {e1}\n → retry without entity")
                    self.wb_run = wandb.init(
                        project=wb.get("project", "doccls"),
                        name=f"{wb.get('run_name_prefix','train')}-fold{fold_id}",
                        tags=wb.get("tags", []),
                        config=cfg,
                        settings=wandb.Settings(start_method="thread")
                    )
                except Exception as e2:
                    print(f"[WARN] wandb second init failed: {e2} → disable W&B")
                    self.wb_enabled = False
                    self.wb_run = None

    def log_params_once(self, params: Dict[str, Any]):
        if self._param_logged: return
        flat = _flatten_dict(params)
        if self.mf_enabled and self.mf_active:
            try: mlflow.log_params(flat)
            except Exception as e: print(f"[WARN] mlflow.log_params failed: {e}")
        if self.wb_enabled and self.wb_run is not None:
            try: self.wb_run.config.update(params, allow_val_change=True)
            except Exception as e: print(f"[WARN] wandb.config.update failed: {e}")
        self._param_logged = True

    def watch_model(self, model, log="gradients", log_freq=200):
        if self.wb_enabled and self.wb_run is not None:
            try: wandb.watch(model, log=log, log_freq=log_freq)
            except Exception as e: print(f"[WARN] wandb.watch failed: {e}")

    def log(self, metrics: Dict[str, Any], step: int | None = None):
        if step is None:
            self._gstep += 1
            step = self._gstep
        if self.wb_enabled and self.wb_run is not None:
            try: wandb.log(metrics, step=step)
            except Exception as e: print(f"[WARN] wandb.log failed: {e}")
        if self.mf_enabled and self.mf_active:
            for k, v in metrics.items():
                try:
                    if hasattr(v, "item"): v = v.item()
                    v = float(v)
                    mlflow.log_metric(k, v, step=step)
                except Exception: pass

    def log_artifact(self, path: str):
        if self.mf_enabled and self.mf_active:
            try:
                if os.path.isdir(path): mlflow.log_artifacts(path)
                else: mlflow.log_artifact(path)
            except Exception as e: print(f"[WARN] mlflow.log_artifact failed: {e}")
        if self.wb_enabled and self.wb_run is not None:
            try:
                if os.path.isdir(path):
                    for root, _, files in os.walk(path):
                        for f in files:
                            wandb.save(os.path.join(root, f), base_path=path)
                else:
                    try:
                        art = wandb.Artifact(os.path.basename(path), type="artifact")
                        art.add_file(path)
                        self.wb_run.log_artifact(art)
                    except Exception:
                        wandb.save(path)
            except Exception as e: print(f"[WARN] wandb artifact logging failed: {e}")

    def finish(self):
        if self.wb_enabled and self.wb_run is not None:
            try: wandb.finish()
            except Exception: pass
        if self.mf_enabled and self.mf_active:
            try: mlflow.end_run()
            except Exception: pass
            self.mf_active = False
