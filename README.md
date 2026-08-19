<h3 align="center">
  <a href="https://arxiv.org/abs/2501.15187">Uni-Sign: Toward Unified Sign Language Understanding at Scale</a>
</h3>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2501.15187-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2501.15187)
[![CSL-News](https://img.shields.io/badge/HuggingFace-CSL%20News-blue.svg)](https://huggingface.co/datasets/ZechengLi19/CSL-News)
[![CSL-News](https://img.shields.io/badge/BaiDu-CSL%20News-green.svg)](https://pan.baidu.com/s/17W6kIreNMHYtD4y2llKmDg?pwd=ncvo)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-ms-asl)](https://paperswithcode.com/sota/sign-language-recognition-on-ms-asl?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-wlasl100)](https://paperswithcode.com/sota/sign-language-recognition-on-wlasl100?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-wlasl-2000)](https://paperswithcode.com/sota/sign-language-recognition-on-wlasl-2000?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-csl-daily)](https://paperswithcode.com/sota/sign-language-recognition-on-csl-daily?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/gloss-free-sign-language-translation-on-csl)](https://paperswithcode.com/sota/gloss-free-sign-language-translation-on-csl?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/gloss-free-sign-language-translation-on-2)](https://paperswithcode.com/sota/gloss-free-sign-language-translation-on-2?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/gloss-free-sign-language-translation-on-3)](https://paperswithcode.com/sota/gloss-free-sign-language-translation-on-3?p=uni-sign-toward-unified-sign-language)

</h5>

![Uni-Sign framework](docs/framework.png)

Uni-Sign is a unified framework for sign language translation (SLT), isolated sign language recognition (ISLR), and continuous sign language recognition (CSLR). It combines pose and optional RGB inputs with spatial-temporal graph convolutional encoders and an mT5 language model. The [original Uni-Sign paper](https://openreview.net/forum?id=0Xt7uT04cQ) was accepted at ICLR 2025.

This repository is an extended fork of the [original Uni-Sign repository](https://github.com/ZechengLi19/Uni-Sign). Its architecture, released checkpoints, and CSL-News resources remain the foundation of this project.

## Contributions in This Fork

Compared with the original repository, this fork includes:

- **Data configuration and loading:** one JSON-configured dataset pipeline for YTASL-style local keypoint JSON, Isharah keypoint JSON, original Uni-Sign pickle data, and CSL-News.
- **Additional dataset support:** the published YTASL keypoint dataset, Isharah data, and compatible per-clip JSON conversions of datasets such as How2Sign, Phoenix, and WLASL.
- **Dataset composition:** mixed-dataset training with natural size-proportional sampling by default and optional dataset-level sampling ratios in both single- and multi-GPU training.
- **Pose and model configuration:** multiple landmark layouts, sign-space normalization, configurable graph topology, register tokens with selectable positions, and optional non-adaptive GCN adjacency.
- **Training and checkpoints:** improved single- and multi-GPU execution, distributed evaluation and metric synchronization, weights-only `--finetune`, full-state DeepSpeed `--resume`, and automatic old-checkpoint cleanup.
- **Evaluation:** deterministic dev/test frame selection, standardized SacreBLEU and ROUGE-L, compatibility with the original metrics, multi-reference ROUGE-L, text normalization, qualitative previews, and JSON result export.
- **Experiment tracking and compatibility:** expanded Weights & Biases logging, optimizer-step tracking, timing, run/output naming, cluster job identifiers, an updated environment guide, and NumPy 2 compatibility.

## Using This Repository

### Documentation

1. [Install and validate the environment](./docs/ENVIRONMENT.md).
2. [Prepare data and choose compatible loaders, layouts, and graphs](./docs/DATASET.md).
3. [Create a JSON data config](./configs/README.md).

Pose extraction and online inference are covered separately in the [demo documentation](./demo/README.md).

### Quick Start

Run commands from the repository root because relative paths in data configs are resolved from the current working directory. After preparing the environment and data, start a single-GPU SLT run with:

```bash
deepspeed --include localhost:0 fine_tuning.py \
  --data_config configs/ytasl.json \
  --task SLT \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --lr 3e-4 \
  --output_dir out/ytasl
```

For multiple GPUs, list them in the launcher, for example `--include localhost:0,1,2,3`. Add `--finetune PATH` to initialize model weights from a `.pth` file or DeepSpeed checkpoint directory. Use `--resume PATH` to restore the complete DeepSpeed training state instead.

The mT5 model and tokenizer are loaded from `./pretrained_weight/mt5-base` by default. Use `--mt5_path PATH_OR_MODEL_ID` to select another local directory or Hugging Face model identifier.

Evaluation-only run:

```bash
deepspeed --include localhost:0 fine_tuning.py \
  --data_config configs/ytasl.json \
  --task SLT \
  --eval \
  --finetune out/ytasl/best_checkpoint.pth \
  --output_dir out/ytasl_eval
```

Evaluation processes every non-null `dev` and `test` split in the selected config. Metrics and up to 100 prediction/reference examples are saved as `<split>_results.json` in `--output_dir`.

## Upstream Uni-Sign Resources

- [Uni-Sign paper](https://arxiv.org/abs/2501.15187)
- [Original repository](https://github.com/ZechengLi19/Uni-Sign)
- [CSL-News dataset](https://huggingface.co/datasets/ZechengLi19/CSL-News)
- [Released checkpoints and pose data](https://huggingface.co/ZechengLi19/Uni-Sign)

For the original pre-training implementation, see [upstream issue #15](https://github.com/ZechengLi19/Uni-Sign/issues/15). Thanks to [@williams-bert](https://github.com/williams-bert) for sharing that implementation.

## Acknowledgements

The original Uni-Sign codebase is adapted from [GFSLT-VLP](https://github.com/zhoubenjia/GFSLT-VLP), while the pose and temporal encoders are derived from [CoSign](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.pdf).

The project also builds on:

- [SSVP-SLT](https://github.com/facebookresearch/ssvp_slt)
- [MMPose](https://github.com/open-mmlab/mmpose)
- [FunASR](https://github.com/modelscope/FunASR)

## Contact

For questions about the original Uni-Sign project, contact Zecheng Li at lizecheng19@gmail.com.

## Citation

If you use Uni-Sign, cite the original paper:

```bibtex
@article{li2025uni,
  title={Uni-Sign: Toward Unified Sign Language Understanding at Scale},
  author={Li, Zecheng and Zhou, Wengang and Zhao, Weichao and Wu, Kepeng and Hu, Hezhen and Li, Houqiang},
  journal={arXiv preprint arXiv:2501.15187},
  year={2025}
}
```
