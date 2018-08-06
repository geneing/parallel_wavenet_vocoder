# coding: utf-8
"""
Synthesis waveform from trained WaveNet.

usage: synthesis.py [options] <checkpoint> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    --conditional=<p>                 Conditional features path.
    --symmetric-mels                  Symmetric mel.
    --max-abs-value=<N>               Max abs value [default: -1].
    --file-name-suffix=<s>            File name suffix [default: ].
    --speaker-id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
import time
from os.path import dirname, join, basename, splitext
import torch
import numpy as np
from nnmnkwii import preprocessing as P
from keras.utils import np_utils
from tqdm import tqdm
import librosa

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

import audio
from hparams import hparams


torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def _to_numpy(x):
    # this is ugly
    if x is None:
        return None
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return x
    # remove batch axis
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.numpy()


def wavegen(model, c=None, g=None):
    """Generate waveform samples by WaveNet.

    Args:
        model (nn.Module) : WaveNet decoder
        c (numpy.ndarray): Conditional features, of shape T x C
        g (scaler): Speaker ID
    Returns:
        numpy.ndarray : Generated waveform samples
    """
    from train import sanity_check
    sanity_check(model, c, g)

    c = _to_numpy(c)
    g = _to_numpy(g)

    model.eval()
    model.make_generation_fast_()

    assert c is not None
    # (Tc, D)
    if c.ndim != 2:
        raise RuntimeError(
            "Expected 2-dim shape (T, {}) for the conditional feature, but {} was actually given.".format(hparams.cin_channels, c.shape))
        assert c.ndim == 2
    Tc = c.shape[0]
    upsample_factor = audio.get_hop_size()
    # Overwrite length according to feature size
    length = Tc * upsample_factor
    # (Tc, D) -> (Tc', D)
    # Repeat features before feeding it to the network
    if not hparams.upsample_conditional_features:
        c = np.repeat(c, upsample_factor, axis=0)

    # B x C x T
    c = torch.FloatTensor(c.T).unsqueeze(0)

    g = None if g is None else torch.LongTensor([g])

    # Transform data to GPU
    dist = torch.distributions.normal.Normal(loc=0., scale=1.)
    z = dist.sample((1, 1, length)).to(device)

    z = z.to(device)
    g = None if g is None else g.to(device)
    c = None if c is None else c.to(device)

    with torch.no_grad():
        start_time = time.time()
        y_hat, _, _, _ = model(x=z, c=c, g=g, device=device, log_scale_min=hparams.log_scale_min)
        duration = time.time() - start_time
        print('Time Evaluation: Generation of {} audio samples took {:.3f} sec ({:.3f} samples/sec)'.format(
            length, duration, length / duration))
        y_hat = y_hat.view(-1).cpu().data.numpy()

    return y_hat


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    dst_dir = args["<dst_dir>"]

    conditional_path = args["--conditional"]
    # From https://github.com/Rayhane-mamah/Tacotron-2
    symmetric_mels = args["--symmetric-mels"]
    max_abs_value = float(args["--max-abs-value"])

    file_name_suffix = args["--file-name-suffix"]
    output_html = args["--output-html"]
    speaker_id = args["--speaker-id"]
    speaker_id = None if speaker_id is None else int(speaker_id)
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    # Load conditional features
    assert checkpoint_path is not None
    c = np.load(conditional_path)
    if c.shape[1] != hparams.num_mels:
        np.swapaxes(c, 0, 1)
    if max_abs_value > 0:
        min_, max_ = 0, max_abs_value
        if symmetric_mels:
            min_ = -max_
        print("Normalize features to desired range [0, 1] from [{}, {}]".format(min_, max_))
        c = np.interp(c, (min_, max_), (0, 1))

    wav_id = conditional_path.split("/")[-1].split('.')[0].replace("mel", "syn_iaf")

    from train_student import build_model

    # Model
    model = build_model(name='student').to(device)

    # Load checkpoint
    print("Load checkpoint from {}".format(checkpoint_path))
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(os.path.join(dst_dir, checkpoint_name), exist_ok=True)

    dst_wav_path = join(os.path.join(dst_dir, checkpoint_name), "{}{}.wav".format(wav_id, file_name_suffix))

    # DO generate
    waveform = wavegen(model, c=c, g=speaker_id)

    # save
    librosa.output.write_wav(dst_wav_path, waveform, sr=hparams.sample_rate)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
