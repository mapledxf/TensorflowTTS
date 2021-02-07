#!/bin/bash
stage=$1

out_dir=/data/xfding/train_result/tts/baker
#Data preparation
if [ $stage -le 1 ]; then
tensorflow-tts-preprocess \
  --rootdir /data/xfding/share/TTS/csmsc \
  --outdir $out_dir \
  --config preprocess/baker_preprocess.yaml \
  --dataset baker
fi
echo -e "$0: stage 1 data preparation completed\n"

#Data normalize
if [ $stage -le 2 ]; then
tensorflow-tts-normalize \
  --rootdir $out_dir \
  --outdir $out_dir \
  --config preprocess/baker_preprocess.yaml \
  --dataset baker
fi
