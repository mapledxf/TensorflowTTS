#!/bin/bash
stage=$1
out_dir=/data/xfding/train_result/tts/baker

#Tacotron2
if [ $stage -le 1 ]; then
python examples/tacotron2/train_tacotron2.py \
  --train-dir $out_dir/train/ \
  --dev-dir $out_dir/valid/ \
  --outdir $out_dir/train.tacotron2.v1/ \
  --config /data/xfding/TensorflowTTS/examples/tacotron2/conf/tacotron2.baker.v1.yaml \
  --use-norm 1 \
  --mixed_precision 0 \
  --resume ""
fi
echo -e "$0: train tacotron2 completed\n"

#Extract duration
if [ $stage -le 2 ]; then
python examples/tacotron2/extract_duration.py \
  --rootdir $out_dir/valid/ \
  --outdir $out_dir/valid/durations/ \
  --checkpoint $out_dir/train.tacotron2.v1/checkpoints/model-200000.h5 \
  --use-norm 1 \
  --config /data/xfding/TensorflowTTS/examples/tacotron2/conf/tacotron2.baker.v1.yaml \
  --batch-size 64 \
  --win-front 3 \
  --win-back 3

python examples/tacotron2/extract_duration.py \
  --rootdir $out_dir/train/ \
  --outdir $out_dir/train/durations/ \
  --checkpoint $out_dir/train.tacotron2.v1/checkpoints/model-200000.h5 \
  --use-norm 1 \
  --config /data/xfding/TensorflowTTS/examples/tacotron2/conf/tacotron2.baker.v1.yaml \
  --batch-size 64 \
  --win-front 3 \
  --win-back 3
fi
echo -e "$0: extract duration completed\n"

#Train fastspeech2
if [ $stage -le 3 ]; then
python examples/fastspeech2/train_fastspeech2.py \
  --train-dir $out_dir/train/ \
  --dev-dir $out_dir/valid/ \
  --outdir $out_dir/train.fastspeech2.v1/ \
  --config ./examples/fastspeech2/conf/fastspeech2.baker.v2.yaml \
  --use-norm 1 \
  --f0-stat $out_dir/stats_f0.npy \
  --energy-stat $out_dir/stats_energy.npy \
  --mixed_precision 1 \
  --resume ""
fi
echo -e "$0: train fastspeech2 completed\n"
