IMG_DB = imgdb-new
TEXT_DB = txtdb-new
VIZ_NUM = 200

# GT_MODEL_DIR = finetune-contrastive-one-to-one-feb
# GT_CKPT = 50000

GT_MODEL_DIR = finetune-contrastive-rest-slow-feb
# # 42.63
GT_CKPT = 18500
# 41.59
# GT_CKPT = 45500

# # 37
# GT_MODEL_DIR = finetune-contrastive-interactive-feb
# GT_CKPT = 36000

# GT_MODEL_DIR = finetune-contrastive-begin-rest-feb
# # 35.16
# GT_CKPT = 20000
# # 32.43%
# # GT_CKPT = 50000

# GT_MODEL_DIR = finetune-new
# GT_CKPT = 50000

# M_MODEL_DIR = finetune-match3
# M_CKPT = 50000

# GT_MODEL_DIR = finetune-contrastive-march_begin_rest_3_loss
# # 39.15%
# # GT_CKPT = 20000
# # 33.3%
# GT_CKPT = 50000

# GT_MODEL_DIR = finetune-contrastive-march_rest_3_loss
# # 39.15%
# # GT_CKPT = 38000
# # 39.32%
# GT_CKPT = 48000

infer_gt: 
	python infer.py  \
		--txt-db $(TEXT_DB) \
		--img-db $(IMG_DB) \
		--model-dir $(GT_MODEL_DIR) \
		--ckpt $(GT_CKPT) \
		--vis-num $(VIZ_NUM) \
		--split test \
		--eval-output-name eval_test \
		--batch_size 1024 \

infer_matching: 
	python infer.py  \
		--txt-db $(TEXT_DB) \
		--img-db $(IMG_DB) \
		--model-dir $(M_MODEL_DIR) \
		--ckpt $(M_CKPT) \
		--vis-num $(VIZ_NUM) \
		--split test \
		--eval-output-name eval_match_test \
		--batch_size 1024 \