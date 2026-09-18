"""Optional platforms are imported only when explicitly enabled."""

from contextlib import ExitStack

from core.config import Config


class PlatformError(Exception):
    """Platform errors must never be tolerated by a tuning study."""


class Platforms:
    def __init__(self, config: Config, directory, run_id: str):
        self.stack = ExitStack()
        self.writers = []
        self.pending = {}
        self.step = None
        self.status = "failed"  # Initialization may fail after opening one platform.
        self.error = None
        try:
            if config.platforms.tensorboard:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter(str(directory / "tensorboard"))
                self.stack.callback(writer.close)
                self.writers.append(
                    lambda values, step: [
                        writer.add_scalar(k, v, step) for k, v in values.items()
                    ]
                )
            if config.platforms.wandb is not None:
                import wandb

                run = wandb.init(
                    config=config.model_dump(), name=run_id, **config.platforms.wandb
                )
                self.stack.callback(self.finish_wandb, run)
                self.writers.append(lambda values, step: run.log(values, step=step))
            if config.platforms.swanlab is not None:
                import swanlab

                swanlab.init(
                    config=config.model_dump(),
                    experiment_name=run_id,
                    **config.platforms.swanlab,
                )
                self.stack.callback(
                    lambda: swanlab.finish(
                        state="crashed" if self.status == "failed" else "success",
                        error=self.error,
                    )
                )
                self.writers.append(lambda values, step: swanlab.log(values, step=step))
        except Exception as exc:
            self.error = str(exc)
            try:
                self.stack.close()
            except Exception as close_exc:
                raise PlatformError(str(close_exc)) from exc
            raise PlatformError(str(exc)) from exc

    def log(self, values, step):
        # Publish one combined point per coordinate: W&B cannot append to a committed step.
        if self.step is not None and step != self.step:
            self.flush()
        self.step = step
        self.pending.update(values)

    def flush(self):
        values, self.pending = self.pending, {}
        try:
            for writer in self.writers:
                if values:
                    writer(values, self.step)
        except Exception as exc:
            raise PlatformError(str(exc)) from exc

    def finish_wandb(self, run):
        # W&B finish has no error-text argument; keep it in the run summary.
        try:
            run.summary["research_status"] = self.status
            if self.error:
                run.summary["research_error"] = self.error
        finally:
            run.finish(exit_code=1 if self.status == "failed" else 0)

    def close(self, status="completed", error=None):
        self.status, self.error = status, error
        try:
            try:
                self.flush()
            except Exception as exc:
                self.status, self.error = "failed", str(exc)
                raise
            finally:
                self.stack.close()
        except Exception as exc:
            raise PlatformError(str(exc)) from exc
