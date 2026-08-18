# Legacy Training Scripts

The shell scripts in this directory are retained from the original Uni-Sign repository as historical reference. They use the original dataset-name and `config.py` workflow and are not kept in sync with the configured dataset interface in `fine_tuning.py`.

In particular, `train_stage1.sh` and `train_stage2.sh` reference `pre_training.py`, which is not included in this repository checkout. The scripts may therefore require an older commit or manual adaptation before use.

For current fine-tuning and evaluation, use `fine_tuning.py --data_config <config.json>`. See the repository dataset documentation and the examples under `configs/` for the supported configuration structure.
