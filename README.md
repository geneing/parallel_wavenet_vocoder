# Parallel WaveNet vocoder

> Note: the code is adapted from [r9y9's wavenet vocoder](https://github.com/r9y9/wavenet_vocoder), u can get more information about wavenet at there.

## [Samples](https://soundcloud.com/dpm1b9vjiaap/sets/parallel-new-samples)
some problems still exists:
1. the generated wav from teacher will have some noise in silence area(1000k step)
2. the generated wav from student still have little noise, but most high frequence noise have been removed

## important details
- use relu rather than leaky relu
- don't apply skip connection after the residual connection, the same as r9y9's implemention
- you should set `share_upsample_conv=True` in `hparams.py` when u train the student


## Quick Start

### Prepare Data
```
python preprocess.py \
    ljspeech \  # data name, i use ljspeech as defalut
    your_data_dir \
    the_dir_to_save_data/\
    --preset=presets/ljspeech_gaussian.json \
```

### Train Autoregressive WaveNet(Teacher)
```
python train.py \
    --preset=presets/ljspeech_gaussian.json \
    --data-root=your_data_dir \
    --hparams='batch_size=9,' \  # in my expreiment, i use 3 gpus(1080Ti)
    --checkpoint-dir=checkpoint-ljspeech \
    --log-event-path=log-ljspeech
```

### Synthesis Using Teacher
```
python synthesis.py \
    --conditional your_local_condition_path \
    --preset=presets/ljspeech_gaussian.json \
    your_teacher_checkpoint_path \
    your_save_dir
```

### Train Distillation WaveNet(Student)
```
python train_student.py \
    --preset=presets/ljspeech_gaussian.json \
    --data-root=your_data_dir \
    --hparams='batch_size=8,' \  # in my expreiment, i use 4 gpus(1080Ti)
    --checkpoint-dir=checkpoint-ljspeech_student \
    --log-event-path=log-ljspeech_student \
    --checkpoint_teacher=your_teacher_checkpoint_path
```

### Synthesis Using Student
```
python synthesis_student.py \
    --conditional your_local_condition_path \
    --preset=presets/ljspeech_gaussian.json \
    your_checkpoint_path \
    your_save_dir
```

## References
+ [ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech](http://export.arxiv.org/pdf/1807.07281)

