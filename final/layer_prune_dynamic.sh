dir_model=models/asr_dynamic_depth/
model_name=10epoch.pth
common_args=(--lm_weight 0 --beam_size 1 --path_model ${dir_model}/${model_name} --n_sample 250 )

LOGLEVEL=WARN python layer_prune_random.py ${common_args[@]} --n_layer 6 | tee ${dir_model}/layer-prune-6.log

# LOGLEVEL=WARN python layer_prune_random.py ${common_args[@]} | tee ${dir_model}/layer-prune-9.log

