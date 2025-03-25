<p align="center">
<!-- <h1 align="center">InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion</h1> -->
<h1 align="center"><img src="assets/logo.png" align="center" width=4% > <strong>InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions</strong></h1>
  <p align="center">
    <a href='https://sirui-xu.github.io/' target='_blank'>Sirui Xu</a><sup><img src="assets/Illinois.jpg" align="center" width=2% ></sup>&emsp;
    <a href='https://hungyuling.com/' target='_blank'>Hung Yu Ling</a><sup><img src="assets/Electronic-Arts-Logo.png" align="center" width=2% ></sup>&emsp;
    <a href='https://yxw.web.illinois.edu/' target='_blank'>Yu-Xiong Wang</a><sup><img src="assets/Illinois.jpg" align="center" width=2% ></sup>&emsp;
    <a href='https://lgui.web.illinois.edu/' target='_blank'>Liang-Yan Gui</a><sup><img src="assets/Illinois.jpg" align="center" width=2% ></sup>&emsp;
    <br>
    <sup><img src="assets/Illinois.jpg" align="center" width=2% ></sup>University of Illinois Urbana Champaign, <sup><img src="assets/Electronic-Arts-Logo.png" align="center" width=2% ></sup>Electronic Arts
    <br>
    <strong>CVPR 2025</strong>
  </p>
</p>

</p>
<p align="center">
  <a href='https://arxiv.org/abs/2502.20390'>
    <img src='https://img.shields.io/badge/Arxiv-2502.20390-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/pdf/2502.20390.pdf'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
  <a href='https://sirui-xu.github.io/InterMimic/'>
  <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
  <a href='https://youtu.be/ZJT387dvI9w'>
  <img src='https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323'></a>
  <a href='https://www.bilibili.com/video/BV1nW9KYFEUX/'>
    <img src='https://img.shields.io/badge/Bilibili-Video-4EABE6?style=flat&logo=Bilibili&logoColor=4EABE6'></a>
  <a href='https://github.com/Sirui-Xu/InterMimic'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=Sirui-Xu.InterMimic&left_color=gray&right_color=orange">
  </a>
</p>

## 🏠 About
<div style="text-align: center;">
    <img src="assets/teaser.png" width=100% >
</div>
Achieving realistic simulations of humans interacting with a wide range of objects has long been a fundamental goal. Extending physics-based motion imitation to complex human-object interactions (HOIs) is challenging due to intricate human-object coupling, variability in object geometries, and artifacts in motion capture data, such as inaccurate contacts and limited hand detail. We introduce InterMimic, a framework that enables a <b>single</b> policy to robustly learn from hours of imperfect MoCap data covering <b>diverse</b> full-body interactions with <b>dynamic and varied</b> objects. Our key insight is to employ a curriculum strategy -- <b>perfect first, then scale up</b>. We first train subject-specific teacher policies to mimic, retarget, and refine motion capture data. Next, we distill these teachers into a student policy, with the teachers acting as online experts providing direct supervision, as well as high-quality references. Notably, we incorporate RL fine-tuning on the student policy to surpass mere demonstration replication and achieve higher-quality solutions. Our experiments demonstrate that InterMimic produces realistic and diverse interactions across multiple HOI datasets. The learned policy <b>generalizes</b> in a zero-shot manner and seamlessly integrates with kinematic generators, elevating the framework from mere imitation to generative modeling of complex human-object interactions.
</br>

## 📹 Demo
<p align="center">
    <img src="assets/InterMimic.gif" align="center" width=70% >
</p>


## 📖 Implementation

### Dependencies

follow the following instructions: 

1. Create new conda environment and install pytroch:

    ```
    conda create -n intermimic python=3.8
    conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
    pip install -r requirement.txt
    ```

    You may also build from [environment.yml](environment.yml), which might contain redundancies,
    ```
    conda env create -f environment.yml
    ```

2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym). 

### 🎓 Teacher Policy Inference


We’ve released a checkpoint for a teacher policy on OMOMO, along with some sample data. To get started:

1. Download the [teacher policy](https://drive.google.com/drive/folders/1biDUmde-h66vUW4npp8FVo2w0wOcK2_k?usp=sharing) and place it under the `teachers/` directory.
2. Then, run the following commands:

    ```bash
    conda activate intermimic
    sh scripts/test.sh
    ```
    
## 🔥 News  
- **[2025-03-25]** We’ve officially released the codebase and checkpoint for teacher policy inference demo — give it a try! ☕️  

## 📝 TODO List  
- [x] Release inference demo for the teacher policy  
- [ ] Release training and inference pipeline for the teacher policy and processed MoCap data  
- [ ] Release student policy distillation training, distilled reference data, and all related checkpoints  
- [ ] Release evaluation pipeline for the student policy  
- [ ] Release all physically simulatable data and processing scripts alongside the [InterAct](https://github.com/wzyabcas/InterAct) launch  
- [ ] Release physics-based text-to-HOI and interaction prediction demo  
- [ ] Add support for Unitree-G1 with dexterous robot hands


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{xu2025intermimic,
  title = {{InterMimic}: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions},
  author = {Xu, Sirui and Ling, Hung Yu and Wang, Yu-Xiong and Gui, Liang-Yan},
  booktitle = {CVPR},
  year = {2025},
}
```

Our data is sourced from **InterAct**. Please consider citing:

```bibtex
@inproceedings{xu2025interact,
  title = {{InterAct}: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation},
  author = {Xu, Sirui and Li, Dongting and Zhang, Yucheng and Xu, Xiyan and Long, Qi and Wang, Ziyin and Lu, Yunzhi and Dong, Shuchang and Jiang, Hezi and Gupta, Akshat and Wang, Yu-Xiong and Gui, Liang-Yan},
  booktitle = {CVPR},
  year = {2025},
}
```
Please also consider citing the specific sub-dataset you used from **InterAct**.

Our integrated kinematic model builds upon **InterDiff**, **HOI-Diff**, and **InterDreamer**. Please consider citing the following if you find this component useful:

```bibtex
@inproceedings{xu2024interdreamer,
  title = {InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction},
  author = {Xu, Sirui and Wang, Ziyin and Wang, Yu-Xiong and Gui, Liang-Yan},
  booktitle = {NeurIPS},
  year = {2024},
}

@inproceedings{xu2023interdiff,
  title = {InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion},
  author = {Xu, Sirui and Li, Zhengyuan and Wang, Yu-Xiong and Gui, Liang-Yan},
  booktitle = {ICCV},
  year = {2023},
}

@article{peng2023hoi,
  title = {HOI-Diff: Text-Driven Synthesis of 3D Human-Object Interactions using Diffusion Models},
  author = {Peng, Xiaogang and Xie, Yiming and Wu, Zizhao and Jampani, Varun and Sun, Deqing and Jiang, Huaizu},
  journal = {arXiv preprint arXiv:2312.06553},
  year = {2023}
}
```

Our SMPL-X-based humanoid model is adapted from PHC. Please consider citing:

```bibtex
@inproceedings{Luo2023PerpetualHC,
  author = {Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
  title = {Perpetual Humanoid Control for Real-time Simulated Avatars},
  booktitle = {ICCV},
  year = {2023}
}
```

## 👏 Acknowledgements and 📚 License

This repository is built on top of the following amazing repositories:

- [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs):
- [PHC](https://github.com/ZhengyiLuo/PHC)
- [PhysHOI](https://github.com/wyhuai/PhysHOI)
- [OMOMO](https://github.com/lijiaman/omomo_release): Major dataset construction
- [InterDiff](https://github.com/Sirui-Xu/InterDiff): Used for kinematic generation
- [HOI-Diff](https://github.com/neu-vi/HOI-Diff): Used for kinematic generation

This code is distributed under an [MIT LICENSE](LICENSE).
Note that our code depends on other libraries and uses datasets which each have their own respective licenses that must also be followed.


## 🌟 Star History

<p align="center">
    <a href="https://star-history.com/#Sirui-Xu/InterMimic&Date" target="_blank">
        <img width="500" src="https://api.star-history.com/svg?repos=Sirui-Xu/InterMimic&type=Date" alt="Star History Chart">
    </a>
<p>
