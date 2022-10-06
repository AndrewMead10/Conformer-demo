from numpy.random import default_rng
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, Union

import torchaudio
import torch
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    extract_archive,
)
from torchaudio import transforms

FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
_CHECKSUMS = {
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz": "743935421bb51cccdb6bdd152e04c5c70274e935c82119ad7faeec31780d811d",  # noqa: E501
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz": "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",  # noqa: E501
}


def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip()))
                       for line in fileobj]
    return output


class SPEECHCOMMANDS(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        subset: Optional[str] = None,
        transform=None,
        num_classes: int = 35,
        noise_max_ratio: float = 0.2,
        time_shift_amount: float = 1
    ) -> None:

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from " +
            "{'training', 'validation', 'testing'}."
        )

        if url in [
            "speech_commands_v0.01",
            "speech_commands_v0.02",
        ]:
            base_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
            ext_archive = ".tar.gz"

            url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, archive, hash_prefix=checksum)
                extract_archive(archive, self._path)

        if subset == "validation":
            self._walker = _load_list(self._path, "validation_list.txt")
        elif subset == "testing":
            self._walker = _load_list(self._path, "testing_list.txt")
        elif subset == "training":
            excludes = set(_load_list(
                self._path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [
                w
                for w in walker
                if HASH_DIVIDER in w and EXCEPT_FOLDER not in w and os.path.normpath(w) not in excludes
            ]
        else:
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [
                w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]

        self.mel = transforms.MelSpectrogram(
            sample_rate=16000, n_mels=40, normalized=True, power=1)

        self.subset = subset

        noise_dir = os.path.join(
            self._path, '_background_noise_')

        noises = sorted(str(p) for p in Path(noise_dir).glob("*.wav"))
        self._noise = [torchaudio.load(noise)[0].squeeze() for noise in noises]
        self.noise_ratios = np.arange(0, noise_max_ratio, 0.05)

        self.rng = default_rng()
        self.time_shift_amount = time_shift_amount

        self.transforms = transform
        self.waveforms = torch.zeros(
            (len(self._walker), 1, 16000))
        self.labels = torch.zeros(len(self._walker))
        self.classes = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
                        'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
        self.i = 0
        for fileid in self._walker:
            relpath = os.path.relpath(fileid, self._path)
            label, _ = os.path.split(relpath)
            if self.label_to_class(label) > num_classes:
                continue
            waveform, _ = torchaudio.load(fileid)
            self.labels[self.i] = self.label_to_class(label)
            self.waveforms[self.i, 0, :waveform.shape[1]] = waveform
            self.i += 1

        self.labels = self.labels.to(torch.long)

    def __getitem__(self, n: int):
        waveform = self.waveforms[n]
        label = self.labels[n]
        waveform = self._augment(waveform)
        return waveform, label

    def _augment(self, waveform):
        if self.subset == 'training':
            waveform = self._time_shift(waveform)
            waveform = self._add_noise(waveform)
            waveform = self.mel(waveform)
            waveform = self.transforms(waveform)
        else:
            waveform = self.mel(waveform)

        return waveform

    def _time_shift(self, waveform):
        shift = self.rng.integers(-1600 * self.time_shift_amount,
                                  1600 * self.time_shift_amount)
        waveform = torch.roll(waveform, shift)
        if shift > 0:
            waveform[0][:shift] = 0
        elif shift < 0:
            waveform[0][shift:] = 0
        return waveform

    def _add_noise(self, waveform):
        randnum = self.rng.integers(58*16000)
        randindex = self.rng.integers(len(self._noise))
        amount = np.random.choice(self.noise_ratios)
        waveform += (amount * self._noise[randindex][randnum:randnum+16000])

        return waveform

    def __len__(self) -> int:
        return self.i

    def label_to_class(self, label):
        return self.classes.index(label)