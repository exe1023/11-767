LOGLEVEL=WARN python head_prune.py \
       --path_model models/truncate/valid.acc.best.pth \
       --skip_layers 2,5,8,11,14,17 | tee models/truncate/head_prune_drop.log
