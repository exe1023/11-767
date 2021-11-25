LOGLEVEL=WARN python head_prune.py \
        --path_model models/interctc_dynamic_depth_finetune/valid.acc.ave_10best.pth \
        --truncate_layer 12 | tee models/interctc_dynamic_depth_finetune/truncate-12.log
exit

LOGLEVEL=WARN python head_prune.py \
        --path_model models/interctc_dynamic_depth_finetune/valid.acc.ave_10best.pth \
        --truncate_layer 6 | tee models/interctc_dynamic_depth_finetune/truncate-6.log

exit

LOGLEVEL=WARN python head_prune.py \
       --path_model models/interctc_dynamic_depth_finetune/valid.acc.ave_10best.pth | tee models/interctc_dynamic_depth_finetune/full.log

