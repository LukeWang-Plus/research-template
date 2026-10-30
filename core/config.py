"""Public configuration only; method parameters remain ordinary dictionaries."""

from pathlib import Path
from typing import Any, Literal

import tomlkit
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Data(Strict):
    name: str
    path: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class Protocol(Strict):
    task: str
    split: str
    metrics: str
    evaluation: str


class Train(Strict):
    epochs: int | None = Field(default=None, ge=1)
    metric: str | None = None
    direction: Literal["min", "max"] | None = None
    keep_top_k: int = Field(default=1, ge=1)
    patience: int | None = Field(default=None, ge=1)
    min_delta: float = Field(default=0.0, ge=0)

    def require(self, *fields):
        missing = [f"train.{name}" for name in fields if getattr(self, name) is None]
        if missing:
            raise ValueError(f"Explicit configuration required: {', '.join(missing)}")


class Platforms(Strict):
    tensorboard: bool = False
    wandb: dict[str, Any] | None = None
    swanlab: dict[str, Any] | None = None


class Storage(Strict):
    remote: str | None = None
    enabled: bool = True


class Pruning(Strict):
    strategy: Literal["none", "median"] = "none"
    params: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolve_params(self):
        if self.strategy == "none":
            if self.params:
                raise ValueError("Pruning strategy 'none' accepts no parameters")
            return self
        from inspect import signature

        from optuna.pruners import MedianPruner

        parameters = signature(MedianPruner).parameters
        unknown = self.params.keys() - parameters.keys()
        if unknown:
            raise ValueError(f"Unknown MedianPruner parameters: {sorted(unknown)}")
        MedianPruner(**self.params)  # Use Optuna's native range validation.
        self.params = {name: p.default for name, p in parameters.items()} | self.params
        return self


class Sampler(Strict):
    strategy: Literal["tpe", "grid"] = "tpe"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolve_params(self):
        if self.strategy == "grid":
            if self.params:
                raise ValueError(
                    "Grid accepts candidates in search.space, not sampler params"
                )
        else:
            from inspect import signature

            from optuna.samplers import TPESampler

            parameters = signature(TPESampler).parameters
            defaults = {name: parameters[name].default for name in TPE.model_fields}
            # TOML has no null: omit native None defaults to retain Optuna's auto mode.
            self.params = TPE.model_validate(defaults | self.params).model_dump(
                exclude_none=True
            )
        return self


class TPE(Strict):
    n_startup_trials: int = Field(ge=0)
    n_ei_candidates: int = Field(ge=1)
    multivariate: bool | None
    group: bool
    constant_liar: bool

    @model_validator(mode="after")
    def check_group(self):
        if self.group and self.multivariate is False:
            raise ValueError("group=true requires multivariate=true")
        return self


class Search(Strict):
    sampler: Sampler = Field(default_factory=Sampler)
    space: dict[str, dict[str, Any]] = Field(default_factory=dict)
    pruning: Pruning = Field(default_factory=Pruning)
    catch: list[str] = Field(default_factory=list)


class Config(Strict):
    method: str
    protocol: Protocol
    data: Data
    params: dict[str, Any] = Field(default_factory=dict)
    train: Train = Field(default_factory=Train)
    platforms: Platforms = Field(default_factory=Platforms)
    storage: Storage | None = None
    search: Search = Field(default_factory=Search)
    seed: int = Field(ge=0, le=2**32 - 1)
    device: str = Field(default="cpu", pattern=r"^(cpu|cuda:[0-9]+)$")
    output: str = "experiments"
    name: str | None = None
    tags: list[str] = Field(default_factory=list)
    note: str = ""
    provenance: dict[str, str | int] = Field(default_factory=dict)


def path(value: str | Path, root: str | Path = ROOT) -> Path:
    value = Path(value)
    return value.resolve() if value.is_absolute() else (Path(root) / value).resolve()


def set_value(data: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def load(
    file: str | Path, overrides: list[str] = (), root: str | Path = ROOT
) -> Config:
    data = tomlkit.parse(path(file, root).read_text(encoding="utf-8")).unwrap()
    for item in overrides:
        key, value = item.split("=", 1)
        set_value(data, key, tomlkit.parse(f"value = {value}").unwrap()["value"])
    return Config.model_validate(data)


def dumps(config: Config) -> str:
    return tomlkit.dumps(config.model_dump(exclude_none=True))


def changed(old: Config, new: Config, allowed: set[str]) -> None:
    a, b = old.model_dump(), new.model_dump()
    for key in allowed:
        for data in (a, b):
            parts = key.split(".")
            current = data
            for part in parts[:-1]:
                current = current[part]
            current.pop(parts[-1], None)
    if a != b:
        raise ValueError(
            f"Configuration changed outside allowed fields: {sorted(allowed)}"
        )
