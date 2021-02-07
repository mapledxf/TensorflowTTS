#!/bin/bash
out_dir=/data/xfding/train_result/tts/baker

# python examples/multiband_melgan/train_multiband_melgan.py \
#   --train-dir $out_dir/train/ \
#   --dev-dir $out_dir/valid/ \
#   --outdir $out_dir/train.multiband_melgan.v1/ \
#   --config ./examples/multiband_melgan/conf/multiband_melgan.baker.v1.yaml \
#   --use-norm 1 \
#   --generator_mixed_precision 1 \
#   --resume ""

python examples/multiband_melgan/train_multiband_melgan.py \
  --train-dir $out_dir/train/ \
  --dev-dir $out_dir/valid/ \
  --outdir $out_dir/train.multiband_melgan.v1/ \
  --config ./examples/multiband_melgan/conf/multiband_melgan.baker.v1.yaml \
  --use-norm 1 \
  --resume $out_dir/train.multiband_melgan.v1/checkpoints/ckpt-200000
echo -e "$0: train multiband melgan completed\n"
