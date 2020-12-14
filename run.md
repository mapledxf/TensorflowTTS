tensorflow-tts-preprocess \
  --rootdir /data/xfding/share/TTS/csmsc \
  --outdir /data/xfding/train_result/baker \
  --config preprocess/baker_preprocess.yaml \
  --dataset baker

tensorflow-tts-normalize \
  --rootdir /data/xfding/train_result/baker \
  --outdir /data/xfding/train_result/baker \
  --config preprocess/baker_preprocess.yaml \
  --dataset baker


CUDA_VISIBLE_DEVICES=1 nohup python examples/tacotron2/train_tacotron2.py \
  --dataset_mapping /data/xfding/train_result/baker/baker_mapper.json \
  --train-dir /data/xfding/train_result/baker/train/ \
  --dev-dir /data/xfding/train_result/baker/valid/ \
  --outdir /data/xfding/train_result/baker/train.tacotron2.v1/ \
  --config /data/xfding/TensorflowTTS/examples/tacotron2/conf/tacotron2.baker.v1.yaml \
  --use-norm 1 \
  --mixed_precision 0 \
  --resume "" > tacotron2.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python examples/tacotron2/extract_duration.py \
  --rootdir /data/xfding/train_result/baker/valid/ \
  --outdir /data/xfding/train_result/baker/valid/durations/ \
  --checkpoint /data/xfding/train_result/baker/train.tacotron2.v1/checkpoints/model-200000.h5 \
  --use-norm 1 \
  --config /data/xfding/TensorflowTTS/examples/tacotron2/conf/tacotron2.baker.v1.yaml \
  --batch-size 64 \
  --win-front 3 \
  --win-back 3 > extract_valid.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python examples/tacotron2/extract_duration.py \
  --rootdir /data/xfding/train_result/baker/train/ \
  --outdir /data/xfding/train_result/baker/train/durations/ \
  --checkpoint /data/xfding/train_result/baker/train.tacotron2.v1/checkpoints/model-200000.h5 \
  --use-norm 1 \
  --config /data/xfding/TensorflowTTS/examples/tacotron2/conf/tacotron2.baker.v1.yaml \
  --batch-size 64 \
  --win-front 3 \
  --win-back 3 > extract_train.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python examples/fastspeech2/train_fastspeech2.py \
  --train-dir /data/xfding/train_result/baker/train/ \
  --dev-dir /data/xfding/train_result/baker/valid/ \
  --outdir /data/xfding/train_result/baker/train.fastspeech2.v1/ \
  --config ./examples/fastspeech2/conf/fastspeech2.baker.v2.yaml \
  --use-norm 1 \
  --f0-stat /data/xfding/train_result/baker/stats_f0.npy \
  --energy-stat /data/xfding/train_result/baker/stats_energy.npy \
  --mixed_precision 1 \
  --resume "" > fastspeech2.log 2>&1 &


==============================================================================================

CUDA_VISIBLE_DEVICES=1 nohup python examples/multiband_melgan/train_multiband_melgan.py \
  --train-dir /data/xfding/train_result/baker/train/ \
  --dev-dir /data/xfding/train_result/baker/valid/ \
  --outdir /data/xfding/train_result/baker/train.multiband_melgan.v1/ \
  --config ./examples/multiband_melgan/conf/multiband_melgan.baker.v1.yaml \
  --use-norm 1 \
  --generator_mixed_precision 1 \
  --resume "" > mb.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/multiband_melgan/train_multiband_melgan.py \
  --train-dir /data/xfding/train_result/baker/train/ \
  --dev-dir /data/xfding/train_result/baker/valid/ \
  --outdir /data/xfding/train_result/baker/train.multiband_melgan.v1/ \
  --config ./examples/multiband_melgan/conf/multiband_melgan.baker.v1.yaml \
  --use-norm 1 \
  --resume /data/xfding/train_result/baker/train.multiband_melgan.v1/checkpoints/ckpt-200000 >> mb.log 2>&1 &

