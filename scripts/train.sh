CUDA_VISIBLE_DEVICES=0 python intermimic/run.py --task InterMimic --headless --wandb \
--cfg_env intermimic/data/cfg/omomo.yaml \
--cfg_train intermimic/data/cfg/train/rlg/omomo.yaml \
--exp_name baseline

CUDA_VISIBLE_DEVICES=1 python intermimic/run.py --task InterMimic --headless --wandb \
--cfg_env intermimic/data/cfg/omomo.yaml \
--cfg_train intermimic/data/cfg/train/rlg/omomo.yaml \
--exp_name ablate_ig_cg

CUDA_VISIBLE_DEVICES=2 python intermimic/run.py --task InterMimic --headless --wandb \
--cfg_env intermimic/data/cfg/omomo.yaml \
--cfg_train intermimic/data/cfg/train/rlg/omomo.yaml \
--exp_name ablate_ig

CUDA_VISIBLE_DEVICES=3 python intermimic/run.py --task InterMimic --headless --wandb \
--cfg_env intermimic/data/cfg/omomo.yaml \
--cfg_train intermimic/data/cfg/train/rlg/omomo.yaml \
--exp_name ablate_cg

CUDA_VISIBLE_DEVICES=4 python intermimic/run.py --task InterMimic --headless --wandb \
--cfg_env intermimic/data/cfg/omomo_rfc.yaml \
--cfg_train intermimic/data/cfg/train/rlg/omomo.yaml \
--exp_name ablate_ig_cg_1N