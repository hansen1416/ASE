Yes. There are several **published pipelines** (and some released code) that explicitly *reuse AMASS motions* while changing the **target body shape** (SMPL/SMPL-X betas), typically with some form of grounding/IK to reduce artefacts:

* **HUMOS (ECCV 2024)** formalizes *shape-conditioned motion* and includes **explicit retargeting baselines** built from AMASS identities:
  **(i)** “TEMOS-Simple”: generate motion on a canonical body, then **naïvely copy target (\beta)** (and gender) from an AMASS identity; **(ii)** a grounded variant; **(iii)** **Rokoko-based retargeting** baseline. 

* **BEDLAM2.0 (Nov 2025)** provides an automated pipeline that **retargets AMASS motions to newly sampled body shapes** (different limb lengths) using **Unreal Engine’s IK Retargeter** with pelvis adjustments to reduce foot sliding; the paper explicitly states they provide the **retarget-to-new-shapes code**. ([arXiv][1])

* **Shape My Moves (CVPR 2025)** is not an “IK retargeter,” but it *operationally* treats shape as first-class: it extracts motion features from SMPL **without canonicalizing everyone to one body**, and it proposes generating **additional SMPL betas** (via Shapy/A2S) to increase shape diversity during training. ([CVF Open Access][2])

* **SMD: Shape Conditioned Human Motion Generation with Diffusion Model (2024)** conditions generation on a **target mesh/shape** (identity features), aiming to produce motion consistent with that shape (often used with AMASS-derived training). ([arXiv][3])

If you just want a *practical* “AMASS-to-new-beta” workflow: many papers’ “naïve retargeting” is literally **swap betas → recompute joints/mesh → fix global translation/grounding**, but high-quality results usually require **contact-aware adjustments (IK / optimization)**—exactly because changing betas changes limb lengths and therefore end-effector trajectories. 

[1]: https://arxiv.org/html/2511.14394v1 "BEDLAM2.0: Synthetic Humans and Cameras in Motion"
[2]: https://openaccess.thecvf.com/content/CVPR2025/papers/Liao_Shape_My_Moves_Text-Driven_Shape-Aware_Synthesis_of_Human_Motions_CVPR_2025_paper.pdf "Shape My Moves: Text-Driven Shape-Aware Synthesis of Human Motions"
[3]: https://arxiv.org/html/2405.06778v1 "Shape Conditioned Human Motion Generation with Diffusion Model"
