IMG_DB = imgdb
TEXT_DB = txtdb-test
VIZ_NUM = 0

GT_MODEL_DIR = finetune-new
GT_CKPT = 50000

M_MODEL_DIR = finetune-match
M_CKPT = 50000

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