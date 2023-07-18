import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np
import soundfile as sf
from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.utils import add_durations, compute_num_samples
from torch.utils.data import DataLoader

from gss.core import GSS, WPE, Activity, Beamformer, Beamformer_With_RefChannel
from gss.utils.data_utils import (
    GssDataset,
    activity_time_to_frequency,
    create_sampler,
    start_end_context_frames,
)

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def get_enhancer_all(
    cuts,
    context_duration=15,  # 15 seconds
    wpe=True,
    wpe_tabs=10,
    wpe_delay=2,
    wpe_iterations=3,
    wpe_psd_context=0,
    activity_garbage_class=True,
    stft_size=1024,
    stft_shift=256,
    stft_fading=True,
    bss_iterations=20,
    bss_iterations_post=1,
    bf_drop_context=True,
    postfilter=None,
    dtype=cp.float32,
):
    assert wpe is True or wpe is False, wpe
    assert len(cuts) > 0

    sampling_rate = cuts[0].recording.sampling_rate

    return Enhancer(
        context_duration=context_duration,
        wpe_block=WPE(
            taps=wpe_tabs,
            delay=wpe_delay,
            iterations=wpe_iterations,
            psd_context=wpe_psd_context,
        )
        if wpe
        else None,
        activity=Activity(
            garbage_class=activity_garbage_class,
            cuts=cuts,
        ),
        gss_block=GSS(
            iterations=bss_iterations,
            iterations_post=bss_iterations_post,
            dtype=dtype,
        ),
        bf_drop_context=bf_drop_context,
        bf_block=Beamformer_With_RefChannel(
            postfilter=postfilter,
        ),
        stft_size=stft_size,
        stft_shift=stft_shift,
        stft_fading=stft_fading,
        sampling_rate=sampling_rate,
    )


@dataclass
class Enhancer:
    """
    This class creates enhancement context (with speaker activity) for the sessions, and
    performs the enhancement.
    """

    wpe_block: WPE
    activity: Activity
    gss_block: GSS
    bf_block: Beamformer_With_RefChannel

    bf_drop_context: bool

    stft_size: int
    stft_shift: int
    stft_fading: bool

    context_duration: float  # e.g. 15
    sampling_rate: int

    def stft(self, x):
        from gss.core.stft_module import stft

        return stft(
            x,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=self.stft_fading,
        )

    def istft(self, X):
        from gss.core.stft_module import istft

        return istft(
            X,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=self.stft_fading,
        )

    def enhance_cuts(
        self,
        cuts,
        exp_dir,
        max_batch_duration=None,
        max_batch_cuts=None,
        num_buckets=2,
        num_workers=1,
        force_overwrite=False,
        only_wpe=False,
    ):
        """
        Enhance the given CutSet.
        """
        num_error = 0

        # Create the dataset, sampler, and data loader
        gss_dataset = GssDataset(
            context_duration=self.context_duration, activity=self.activity
        )
        gss_sampler = create_sampler(
            cuts,
            max_duration=max_batch_duration,
            max_cuts=max_batch_cuts,
            num_buckets=num_buckets,
        )
        dl = DataLoader(
            gss_dataset,
            sampler=gss_sampler,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=False,
        )

        def _save_worker(orig_cuts, x_hats, recording_id, speaker, ref_channel, target_speaker_idx, failed):
            # x_hat: (n_srcs, n_samples)
            out_dir = exp_dir / recording_id
            metadatas = []
            offset = 0
            for cut in orig_cuts:
                save_paths = {} if only_wpe else []
                base_dir = Path(cut.recording.sources[0].source).parent.parent.parent.parent.parent
                if failed:
                    assert not only_wpe
                    x_hats = np.tile(x_hats, (5, 1))
                for n, x_hat in enumerate(x_hats):
                    idx = cut.channel[n] if only_wpe else n
                    save_path = Path(
                        f"{recording_id}-{speaker}-{round(100*cut.start):06d}_{round(100*cut.end):06d}_{idx}.flac"
                    )
                    if force_overwrite or not (out_dir / save_path).exists():
                        st = compute_num_samples(offset, self.sampling_rate)
                        en = st + compute_num_samples(cut.duration, self.sampling_rate)
                        x_hat_cut = x_hat[None, st:en]
                        logging.debug("Saving enhanced signal")
                        sf.write(
                            file=str(out_dir / save_path),
                            data=x_hat_cut.transpose(),
                            samplerate=self.sampling_rate,
                            format="FLAC",
                        )
                        if only_wpe:
                            save_paths[idx] = str(base_dir / out_dir / save_path)
                        else:
                            save_paths.append(str(base_dir / out_dir / save_path))
                    else:
                        logging.info(f"File {save_path} already exists. Skipping.")
                # if force_overwrite or not (out_dir / save_path).exists():
                    # Update offset for the next cut
                    # offset = add_durations(
                    #     offset, cut.duration, sampling_rate=self.sampling_rate
                    # )
                assert len(cut.supervisions) == 1
                mixture_paths = {}
                for ch in cut.channel:
                    mixture_info = cut.recording.sources[ch]
                    assert ch == mixture_info.channels[0], (ch, mixture_info[0])
                    # mixture_paths.append(mixture_info.source)
                    mixture_paths[ch] = mixture_info.source
                sampling_rate = cut.recording.sampling_rate
                key = "dereverberated_mixture" if only_wpe else "pseudo_target"
                metadata = {
                    "mixture": mixture_paths,
                    key: save_paths,
                    "channels": cut.channel,
                    "ref_channel": ref_channel,
                    "start": cut.start,
                    "duration": cut.duration,
                    "recording_id": recording_id,
                    "speaker": speaker,
                    "target_speaker_idx": target_speaker_idx,
                    "sampling_rate": sampling_rate,
                    "text": cut.supervisions[0].text,
                    "failed": failed,
                }
                metadatas.append(metadata)
            return metadatas

        # Iterate over batches
        futures = []
        total_processed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for batch_idx, batch in enumerate(dl):
                batch = SimpleNamespace(**batch)
                logging.info(
                    f"Processing batch {batch_idx+1} {batch.recording_id, batch.speaker}: "
                    f"{len(batch.orig_cuts)} segments = {batch.duration}s (total: {total_processed} segments)"
                )
                total_processed += len(batch.orig_cuts)
                out_dir = exp_dir / batch.recording_id
                out_dir.mkdir(parents=True, exist_ok=True)

                file_exists = []
                if not force_overwrite:
                    for cut in batch.orig_cuts:
                        save_path = Path(
                            f"{batch.recording_id}-{batch.speaker}-{round(100*cut.start):06d}_{round(100*cut.end):06d}.flac"
                        )
                        file_exists.append((out_dir / save_path).exists())

                    if all(file_exists):
                        logging.info("All files already exist. Skipping.")
                        continue

                # Sometimes the segment may be large and cause OOM issues in CuPy. If this
                # happens, we increasingly chunk it up into smaller segments until it can
                # be processed without breaking.
                num_chunks = 1
                while True:
                    try:
                        x_hat, ref_channel = self.enhance_batch(
                            batch.audio,
                            batch.activity,
                            batch.speaker_idx,
                            num_chunks=num_chunks,
                            left_context=batch.left_context,
                            right_context=batch.right_context,
                            only_wpe=only_wpe,
                        )
                        failed = False
                        break
                    except cp.cuda.memory.OutOfMemoryError:
                        num_chunks = num_chunks + 1
                        logging.warning(
                            f"Out of memory error while processing the batch. Trying again with {num_chunks} chunks."
                        )
                    except Exception as e:
                        logging.error(f"Error enhancing batch: {e}")
                        num_error += 1
                        # Keep the original signal (only load channel 0)
                        # NOTE (@desh2608): One possible issue here is that the whole batch
                        # may fail even if the issue is only due to one segment. We may
                        # want to handle this case separately.
                        x_hat = batch.audio[0:1].cpu().numpy()
                        ref_channel = int(batch.orig_cuts[0].channel[0])
                        failed = True
                        break

                # Save the enhanced cut to disk
                ref_channel = int(cp.asnumpy(ref_channel))
                futures.append(
                    executor.submit(
                        _save_worker,
                        batch.orig_cuts,
                        x_hat,
                        batch.recording_id,
                        batch.speaker,
                        ref_channel,
                        batch.speaker_idx,
                        failed,
                    )
                )
                # if batch_idx == 2:
                #     break

        out_metadatas = []
        for future in futures:
            metadatas = future.result()
            out_metadatas.extend(metadatas)
        return num_error, out_metadatas


    def enhance_batch(
        self, obs, activity, speaker_id, num_chunks=1, left_context=0, right_context=0, only_wpe=False,
    ):

        logging.debug(f"Converting activity to frequency domain")
        activity_freq = activity_time_to_frequency(
            activity,
            stft_window_length=self.stft_size,
            stft_shift=self.stft_shift,
            stft_fading=self.stft_fading,
            stft_pad=True,
        )

        # Convert to cupy array (putting it on the GPU)
        obs = cp.asarray(obs)

        logging.debug(f"Computing STFT")
        Obs = self.stft(obs)

        D, T, F = Obs.shape

        # Process observation in chunks
        chunk_size = int(np.ceil(T / num_chunks))
        masks = []
        for i in range(num_chunks):
            st = i * chunk_size
            en = min(T, (i + 1) * chunk_size)
            Obs_chunk = Obs[:, st:en, :]

            logging.debug(f"Applying WPE")
            if self.wpe_block is not None:
                Obs_chunk = self.wpe_block(Obs_chunk)
                # Replace the chunk in the original array (to save memory)
                Obs[:, st:en, :] = Obs_chunk

            if not only_wpe:
                logging.debug(f"Computing GSS masks")
                masks_chunk = self.gss_block(Obs_chunk, activity_freq[:, st:en])
                masks.append(masks_chunk)

        if only_wpe:
            obs = self.istft(Obs)
            obs = obs[..., left_context:-right_context]
            return obs, 0

        masks = cp.concatenate(masks, axis=1)
        if self.bf_drop_context:
            logging.debug("Dropping context for beamforming")
            left_context_frames, right_context_frames = start_end_context_frames(
                left_context,
                right_context,
                stft_size=self.stft_size,
                stft_shift=self.stft_shift,
                stft_fading=self.stft_fading,
            )
            logging.debug(
                f"left_context_frames: {left_context_frames}, right_context_frames: {right_context_frames}"
            )

            masks[:, :left_context_frames, :] = 0
            if right_context_frames > 0:
                masks[:, -right_context_frames:, :] = 0

        # target_mask = masks[speaker_id]
        # distortion_mask = cp.sum(masks, axis=0) - target_mask

        logging.debug("Applying beamforming with computed masks")
        X_hats = []
        # assert num_chunks == 1, "Currently, this code can work only when num_chunks==1"
        for i in range(num_chunks):
            X_hat = []
            # start and enc of each chunk
            st = i * chunk_size
            en = min(T, (i + 1) * chunk_size)
            # first enhance the target signal to obtain reference channel
            target_mask = masks[speaker_id]
            distortion_mask = cp.sum(masks, axis=0) - target_mask
            X_hat_target, ref_channel = self.bf_block(
                Obs[:, st:en, :],
                target_mask=target_mask[st:en],
                distortion_mask=distortion_mask[st:en],
                ref_channel=None,
            )
            # enhance each source
            for n, target_mask in enumerate(masks):
                assert ref_channel is not None, ref_channel
                # target speaker is already enhanced
                if n == speaker_id:
                    X_hat.append(X_hat_target)
                    continue
                distortion_mask = cp.sum(masks, axis=0) - target_mask
                X_hat_interference, _ = self.bf_block(
                    Obs[:, st:en, :],
                    target_mask=target_mask[st:en],
                    distortion_mask=distortion_mask[st:en],
                    ref_channel=ref_channel,
                )
                X_hat.append(X_hat_interference)
            X_hat = cp.stack(X_hat, axis=0) # X_hat: (n_srcs, n_frames, n_freqs)
            X_hats.append(X_hat)

        # X_hat = cp.stack(X_hat, axis=0) # X_hat: (n_srcs, n_frames, n_freqs)
        X_hats = cp.concatenate(X_hat, axis=1) #(n_srcs, n_frames, n_freqs))

        logging.debug("Computing inverse STFT")
        x_hat = self.istft(X_hat)  # returns a numpy array

        if x_hat.ndim == 1:
            x_hat = x_hat[np.newaxis, :]

        # Trim x_hat to original length of cut
        x_hat = x_hat[..., left_context:-right_context, :]

        return x_hat, ref_channel


def metadata_collector(
    cuts,
    context_duration=15,  # 15 seconds
    wpe=True,
    wpe_tabs=10,
    wpe_delay=2,
    wpe_iterations=3,
    wpe_psd_context=0,
    activity_garbage_class=True,
    stft_size=1024,
    stft_shift=256,
    stft_fading=True,
    bss_iterations=20,
    bss_iterations_post=1,
    bf_drop_context=True,
    postfilter=None,
    dtype=cp.float32,
):
    assert wpe is True or wpe is False, wpe
    assert len(cuts) > 0

    sampling_rate = cuts[0].recording.sampling_rate

    return Metadata_Collector(
        context_duration=context_duration,
        wpe_block=WPE(
            taps=wpe_tabs,
            delay=wpe_delay,
            iterations=wpe_iterations,
            psd_context=wpe_psd_context,
        )
        if wpe
        else None,
        activity=Activity(
            garbage_class=activity_garbage_class,
            cuts=cuts,
        ),
        gss_block=GSS(
            iterations=bss_iterations,
            iterations_post=bss_iterations_post,
            dtype=dtype,
        ),
        bf_drop_context=bf_drop_context,
        bf_block=Beamformer_With_RefChannel(
            postfilter=postfilter,
        ),
        stft_size=stft_size,
        stft_shift=stft_shift,
        stft_fading=stft_fading,
        sampling_rate=sampling_rate,
    )

@dataclass
class Metadata_Collector:
    """
    This class creates enhancement context (with speaker activity) for the sessions, and
    performs the enhancement.
    """

    wpe_block: WPE
    activity: Activity
    gss_block: GSS
    bf_block: Beamformer_With_RefChannel

    bf_drop_context: bool

    stft_size: int
    stft_shift: int
    stft_fading: bool

    context_duration: float  # e.g. 15
    sampling_rate: int

    def enhance_cuts(
        self,
        cuts,
        exp_dir,
        max_batch_duration=None,
        max_batch_cuts=None,
        num_buckets=2,
        num_workers=1,
        force_overwrite=False,
        only_wpe=False,
    ):
        """
        Enhance the given CutSet.
        """
        num_error = 0

        # Create the dataset, sampler, and data loader
        gss_dataset = GssDataset(
            context_duration=self.context_duration, activity=self.activity
        )
        gss_sampler = create_sampler(
            cuts,
            max_duration=max_batch_duration,
            max_cuts=max_batch_cuts,
            num_buckets=num_buckets,
        )
        dl = DataLoader(
            gss_dataset,
            sampler=gss_sampler,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=False,
        )

        def _save_worker(orig_cuts, recording_id, speaker, target_speaker_idx):
            # x_hat: (n_srcs, n_samples)
            out_dir = exp_dir / recording_id
            metadatas = []
            offset = 0
            for cut in orig_cuts:
                # base_dir = Path(cut.recording.sources[0].source).parent.parent.parent.parent.parent
                base_dir=Path("/ocean/projects/cis210027p/ksaijo/experiments/espnet_chime7_task1/egs2/chime7_task1/asr2")
                save_path = Path(
                    f"{recording_id}-{speaker}-{round(100*cut.start):06d}_{round(100*cut.end):06d}.flac"
                )
                st = compute_num_samples(offset, self.sampling_rate)
                en = st + compute_num_samples(cut.duration, self.sampling_rate)
                logging.debug("Saving enhanced signal")
                save_path = str(base_dir / out_dir / save_path)

                assert len(cut.supervisions) == 1
                mixture_paths = {}
                for ch in cut.channel:
                    mixture_info = cut.recording.sources[ch]
                    assert ch == mixture_info.channels[0], (ch, mixture_info[0])
                    # mixture_paths[ch] = mixture_info.source
                    path_part = mixture_info.source.split("/")[-5:]
                    path_part = "/".join(path_part)
                    mixture_paths[ch] = (str(base_dir / path_part))
                sampling_rate = cut.recording.sampling_rate
                metadata = {
                    "mixture": mixture_paths,
                    "pseudo_target": save_path,
                    "channels": cut.channel,
                    "start": cut.start,
                    "duration": cut.duration,
                    "recording_id": recording_id,
                    "speaker": speaker,
                    "target_speaker_idx": target_speaker_idx,
                    "sampling_rate": sampling_rate,
                    "text": cut.supervisions[0].text,
                }
                metadatas.append(metadata)
            return metadatas

        # Iterate over batches
        futures = []
        total_processed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for batch_idx, batch in enumerate(dl):
                batch = SimpleNamespace(**batch)
                logging.info(
                    f"Processing batch {batch_idx+1} {batch.recording_id, batch.speaker}: "
                    f"{len(batch.orig_cuts)} segments = {batch.duration}s (total: {total_processed} segments)"
                )
                total_processed += len(batch.orig_cuts)
                out_dir = exp_dir / batch.recording_id
                out_dir.mkdir(parents=True, exist_ok=True)

                file_exists = []
                for cut in batch.orig_cuts:
                    save_path = Path(
                        f"{batch.recording_id}-{batch.speaker}-{round(100*cut.start):06d}_{round(100*cut.end):06d}.flac"
                    )
                    file_exists.append((out_dir / save_path).exists())

                # if all(file_exists):
                #     logging.info("All files already exist. OK.")
                # else:
                #     assert False, "something wrong?"

                # Save the enhanced cut to disk
                futures.append(
                    executor.submit(
                        _save_worker,
                        batch.orig_cuts,
                        batch.recording_id,
                        batch.speaker,
                        batch.speaker_idx,
                    )
                )
                # if batch_idx == 2:
                #     break

        out_metadatas = []
        for future in futures:
            metadatas = future.result()
            out_metadatas.extend(metadatas)
        return num_error, out_metadatas
