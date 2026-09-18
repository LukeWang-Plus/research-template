"""CPU/GPU end-to-end smoke with isolated, automatically removed generated data."""

import argparse
import tempfile
from pathlib import Path

import torch

from core.config import ROOT, load
from core.records import read_json
from core.runner import execute, prepare, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(
        f"torch={torch.__version__} CUDA-build={torch.version.cuda} device={args.device}"
    )
    if args.device.startswith("cuda:"):
        print(torch.cuda.get_device_name(torch.device(args.device)))
    with tempfile.TemporaryDirectory(prefix="research-smoke-") as temporary:
        root = Path(temporary)
        for method in ("linear", "ragged"):
            config = load(
                ROOT / "configs" / f"{method}.toml", [f'device="{args.device}"']
            )
            prepare(config, root)
            execute(config, id=method, root=root)
            config.train.epochs += 1
            execute(config, id=method, resume=True, root=root)
            # The runner resolves training settings from the selected checkpoint's launch.
            original = load(root / "experiments/runs" / method / "config.toml")
            test(original, source=method, id=f"{method}-test", root=root)
            result = read_json(
                root / "experiments/runs" / f"{method}-test" / "record.json"
            )
            assert result["status"] == "completed"
            print(f"{method}: train/resume/test passed; {result['result']}")


if __name__ == "__main__":
    main()
