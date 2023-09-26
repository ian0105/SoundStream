import torch
from torch.utils.data import Dataset
import soundfile as sf
from pathlib import Path
import resampy
import numpy as np
from typing import Tuple, Union, Optional, List, Dict, Any
import math
from tqdm import tqdm
import wave

class Base(Dataset):
    def __init__(self,
                 data_list: List[Tuple[Path, int]],
                 sampling_rate: int = None,
                 segment_time: int = 3,
                 **kwargs,
                 ):
        super().__init__()
        boundaries = [0]
        self.data_list = []

        print("Preprocessing data...")
        for filename, sr in tqdm(data_list):
            with wave.open(str(filename), "rb") as audio_file:
                audio_length_frames = audio_file.getnframes()
                sample_rate = audio_file.getframerate()
                audio_length_seconds = audio_length_frames / float(sample_rate)
            num_chunks = math.ceil(audio_length_seconds / segment_time)
            boundaries.append(boundaries[-1] + num_chunks)
            self.data_list.append((filename, sr, segment_time))

        self.boundaries = np.array(boundaries)
        self.segment_length = segment_time * sampling_rate

    def __len__(self) -> int:
        return self.boundaries[-1]

    def _get_file_idx_and_chunk_idx(self, index: int) -> Tuple[int, int]:
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)
        chunk_index = index - self.boundaries[bin_pos]
        return bin_pos, chunk_index

    def _get_waveforms(self, index: int, chunk_index: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get waveform without resampling."""
        wav_file, sr, length_in_time = self.data_list[index]
        offset = int(chunk_index * length_in_time * sr)
        frames = int(length_in_time * sr)

        data, _ = sf.read(
            wav_file, start=offset, frames=frames, dtype='float32', always_2d=True)
        data = data.mean(axis=1, keepdims = False)
        return data

    def __getitem__(self, index: int) -> torch.Tensor:
        file_idx, chunk_idx = self._get_file_idx_and_chunk_idx(index)
        data = self._get_waveforms(file_idx, chunk_idx)

        if data.shape[0] != self.segment_length:
            data = resampy.resample(
                data, data.shape[0], self.segment_length, axis=0, filter='kaiser_fast')[:self.segment_length]
#            if data.shape[0] < self.segment_length:
#                data = np.pad(
#                      data, ((0, self.segment_length - data.shape[0]),), 'constant')
        return torch.tensor(data)