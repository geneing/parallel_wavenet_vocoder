# Parallel WaveNet vocoder

About the samples, i will add later, now the samples is not much good.

> Note: the code is adapted from [r9y9's wavenet vocoder](https://github.com/r9y9/wavenet_vocoder), u can get more information about wavenet at there.

## To Do List

- [x] add gaussian distribution to origin wavenet
- [x] obtain a good single gaussian teacher (430K now)
- [x] add gaussian student
- [ ] obtain a good single gaussian student (training, 70k, i use 330k as the teacher)
- [ ] test share upsample conv(now not share)

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

