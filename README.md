<h3 align="center"><a href="" style="color:#9C276A">
Uni-Sign: Toward Unified Sign Language Understanding at Scale</a></h3>
<h5 align="center"> 
If our project helps you, please give us a star🌟 on GitHub, that would motivate us a lot!
</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2501.15187-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2501.15187) 
[![CSL-Dataset](https://img.shields.io/badge/HuggingFace🤗-%20CSL%20News-blue.svg)](https://huggingface.co/datasets/ZechengLi19/CSL-News)
[![CSL-Dataset](https://img.shields.io/badge/BaiDu☁-%20CSL%20News-green.svg)](https://pan.baidu.com/s/17W6kIreNMHYtD4y2llKmDg?pwd=ncvo) 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-ms-asl)](https://paperswithcode.com/sota/sign-language-recognition-on-ms-asl?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-wlasl100)](https://paperswithcode.com/sota/sign-language-recognition-on-wlasl100?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-wlasl-2000)](https://paperswithcode.com/sota/sign-language-recognition-on-wlasl-2000?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-csl-daily)](https://paperswithcode.com/sota/sign-language-recognition-on-csl-daily?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/gloss-free-sign-language-translation-on-csl)](https://paperswithcode.com/sota/gloss-free-sign-language-translation-on-csl?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/gloss-free-sign-language-translation-on-2)](https://paperswithcode.com/sota/gloss-free-sign-language-translation-on-2?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/gloss-free-sign-language-translation-on-3)](https://paperswithcode.com/sota/gloss-free-sign-language-translation-on-3?p=uni-sign-toward-unified-sign-language)
</h5>

![Uni-Sign](docs/framework.png)

## 💥 News
[2025/1/25] This [paper](https://openreview.net/pdf?id=0Xt7uT04cQ) is accepted by `ICLR 2025` 🎉🎉!

[2025/2/24] Release CSL-News dataset and code implementation.

[2025/6/5] Release lighter pose extraction and online inference implementation. Check it out [here](./demo/README.md) 🎊🎊. 

## 🛠️ Installation
We suggest creating a new conda environment. For current CUDA/ROCm/CPU setup commands, see [docs/ENVIRONMENT.md](./docs/ENVIRONMENT.md).

## 📖 Preparation
Please follow the instructions provided in [DATASET.md](./docs/DATASET.md) for data preparation.

## 🔨 Training & Evaluation
All scripts must be executed within the Uni-Sign directory. 

For the pre-training code, please refer to [Issue #15](https://github.com/ZechengLi19/Uni-Sign/issues/15) for detailed instructions. Thanks to [@williams-bert](https://github.com/williams-bert) for generously sharing the code implementation.
### Training
**Stage 1**: pose-only pre-training.
```bash
bash ./script/train_stage1.sh
```
**Stage 2**: RGB-pose pre-training.
```bash
bash ./script/train_stage2.sh
```
**Stage 3**: downstream fine-tuning.
```bash
bash ./script/train_stage3.sh
```

### Evaluation
After completing stage 3 fine-tuning, performance evaluation on a single GPU can be performed using the following command:
```bash
bash ./script/eval_stage3.sh
```

## 👨‍💻 Todo
- [x] Release CSL-News dataset
- [x] Release Uni-Sign implementation 
- [x] Release pose extraction and online inference implementation

## 📮 Contact
If you have any questions, please feel free to contact Zecheng Li (lizecheng19@gmail.com). Thank you.

## 👍 Acknowledgement
The codebase of Uni-Sign is adapted from [GFSLT-VLP](https://github.com/zhoubenjia/GFSLT-VLP), while the implementations of the pose/temporal encoders are derived from [CoSign](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.pdf). We sincerely appreciate the authors of CoSign for personally sharing their code 🙏. \
We are also grateful for the following projects our Uni-Sign arise from:
* 🤟[SSVP-SLT](https://github.com/facebookresearch/ssvp_slt): a excellent sign language translation framework! 
* 🏃️[MMPose](https://github.com/open-mmlab/mmpose): an open-source toolbox for pose estimation.
* 🤠[FUNASR](https://github.com/modelscope/FunASR): a high-performance speech-to-text toolkit.


## 📑 Citation
If you find Uni-Sign useful for your research and applications, please cite using this BibTeX:
```
@article{li2025uni,
  title={Uni-Sign: Toward Unified Sign Language Understanding at Scale},
  author={Li, Zecheng and Zhou, Wengang and Zhao, Weichao and Wu, Kepeng and Hu, Hezhen and Li, Houqiang},
  journal={arXiv preprint arXiv:2501.15187},
  year={2025}
}
```
