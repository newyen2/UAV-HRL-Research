"""Unified command-line entry point for training methods."""
import importlib
from pathlib import Path

import utils


def main(argv=None):
    """先套用命令列設定與尺寸，再載入所選方法並開始訓練。"""
    args = utils.parse_args("train", argv)
    utils.apply_settings(args.model, args.settings)
    if args.output_dir is not None:
        utils.RECORD_SETTING.PATH = args.output_dir

    from runtime.configuration import METHODS

    module = importlib.import_module(Path(METHODS[args.model]["module"]).stem)
    return module.train()


if __name__ == "__main__":
    main()
