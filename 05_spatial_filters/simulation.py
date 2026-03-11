from pathlib import Path
from functools import reduce
import numpy as np
import mne
from mne import make_ad_hoc_cov
from mne.simulation import simulate_raw, simulate_sparse_stc, add_eog, add_noise

root = Path(__file__).parent


def combine_stcs(*stcs):
    vertices = [
        reduce(np.union1d, [stc.vertices[0] for stc in stcs]),
        reduce(np.union1d, [stc.vertices[1] for stc in stcs]),
    ]
    expanded = [stc.expand(vertices) for stc in stcs]
    return reduce(lambda a, b: a + b, expanded)


class utils:
    combine_stcs = combine_stcs


def alpha_wave(times):
    """Simulate a 10 Hz alpha oscillation."""
    return 1e-7 * np.sin(2 * np.pi * 10 * times)


def theta_wave(times):
    """Simulate a 6 Hz theta oscillation with amplitude modulation."""
    return 1e-7 * (1 + np.sin(2 * np.pi * 1 * times)) * np.sin(2 * np.pi * 6 * times)


def auditory_left(times):
    """Left auditory evoked response: peaks at 100 ms after each 1.0 s stimulus."""
    t_local = (times % 1.0) - 0.1
    return 5e-7 * np.sin(2 * np.pi * 5 * t_local) * np.exp(-(t_local**2) / 0.005)


def auditory_right(times):
    """Right auditory evoked response: peaks at 130 ms after each 1.0 s stimulus."""
    t_local = (times % 1.0) - 0.13
    return 5e-7 * np.sin(2 * np.pi * 5 * t_local) * np.exp(-(t_local**2) / 0.005)


raw_fname = mne.datasets.eegbci.load_data(subjects=1, runs=[6])[0]
raw = mne.io.read_raw_edf(raw_fname, preload=True, verbose=False)
montage = mne.channels.make_standard_montage("standard_1005")
mne.datasets.eegbci.standardize(raw)
raw.set_montage(montage)
raw.resample(100, verbose=False)


fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
src = mne.read_source_spaces(fs_dir / "bem" / "fsaverage-ico-5-src.fif")
bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
fwd = mne.make_forward_solution(
    raw.info, trans="fsaverage", src=src, bem=bem, eeg=True, verbose=False, n_jobs=-1
)
raw.info["dev_head_t"] = fwd["info"]["dev_head_t"]
labels = mne.read_labels_from_annot(
    "fsaverage", parc="aparc", subjects_dir=fs_dir.parent, verbose=False
)

times = np.arange(0, 600, 1 / raw.info["sfreq"])  # 15 min at 100 Hz

label_occ = [l for l in labels if l.name == "pericalcarine-lh"][0]
label_front = [l for l in labels if l.name == "superiorfrontal-lh"][0]
label_aud_l = [l for l in labels if l.name == "transversetemporal-lh"][0]
label_aud_r = [l for l in labels if l.name == "transversetemporal-rh"][0]

stc1 = simulate_sparse_stc(
    fwd["src"],
    n_dipoles=1,
    times=times,
    data_fun=alpha_wave,
    labels=[label_occ],
    location="center",
    random_state=0,
)
stc2 = simulate_sparse_stc(
    fwd["src"],
    n_dipoles=1,
    times=times,
    data_fun=theta_wave,
    labels=[label_front],
    location="center",
    random_state=1,
)
stc3 = simulate_sparse_stc(
    fwd["src"],
    n_dipoles=1,
    times=times,
    data_fun=auditory_left,
    labels=[label_aud_l],
    location="center",
    random_state=2,
)
stc4 = simulate_sparse_stc(
    fwd["src"],
    n_dipoles=1,
    times=times,
    data_fun=auditory_right,
    labels=[label_aud_r],
    location="center",
    random_state=3,
)

stc = utils.combine_stcs(stc1, stc2, stc3, stc4)
raw_sim = simulate_raw(raw.info, stc, forward=fwd, verbose=False)

cov = make_ad_hoc_cov(raw_sim.info, std=dict(eeg=10e-6))
raw_sim = add_noise(raw_sim, cov, random_state=42)
add_eog(raw_sim, random_state=42)
raw_sim.set_eeg_reference("average", projection=True)
raw_sim.filter(1, 40, verbose=False)

sfreq = raw_sim.info["sfreq"]
event_samples = np.arange(0, len(times), int(sfreq * 2.0))
events = np.column_stack(
    [
        event_samples,
        np.zeros(len(event_samples), dtype=int),
        np.ones(len(event_samples), dtype=int),
    ]
).astype(int)

event_id = {"auditory": 1}
epochs = mne.Epochs(
    raw_sim,
    events,
    event_id=event_id,
    tmin=-0.2,
    tmax=0.8,
    baseline=None,
    preload=True,
    verbose=False,
)

raw_sim.save(root / "sim-raw.fif", overwrite=True, verbose=False)
epochs.save(root / "sim-epo.fif", overwrite=True, verbose=False)
stc.save(root / "sim", overwrite=True, verbose=False)
