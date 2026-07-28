# Limitations, deviations, and unblocker

No named dataset was downloaded and no Qwen-3-style or ViT model was trained.
The repository contains no paper training implementation or raw trajectory
files. CPU-only authorization is incompatible with the paper's H100 setup and
the minimum 120 configurations implied by four datasets, three rates, and ten
seeds.

The exact normalized-sigmoid counterexample invalidates one premise used in
the paper's interpretation, but it does not contradict an observed curve on a
specific training trajectory. It is therefore not labeled a falsification of
Claim 5.

Unblocking requires the authors' executable training code and raw trajectories
or authorization and suitable accelerator capacity for the exact campaign.
