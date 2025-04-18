CUDA_VISIBLE_DEVICES=7 python intermimic/run.py --task InterMimic \
--cfg_env intermimic/data/cfg/omomo.yaml \
--cfg_train intermimic/data/cfg/train/rlg/omomo.yaml \
--num_envs 512 \
--headless \
--debug