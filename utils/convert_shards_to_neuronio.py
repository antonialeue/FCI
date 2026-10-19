"""
Convert a sharded FCI simulation dataset into NeuronIO-style pickles, so the ELM code
(elmnetwork/src/datasets/neuronio) can train on it without a new parser.

Source (see utils/simulation_shards.py):

    <models>/<model>/train/shard_000..019.h5     10,000 sims, 500 per shard
    <models>/<model>/test/shard_000..004.h5      2,500 sims
    <models>/<model>/test/summary.pkl            firing_rate_information['normalized_test_set_by_index']

Output, one pickle per SIMS_PER_FILE simulations:

    <out_dir>/<model>/train/            train shard_000..018   (9,500 sims)
    <out_dir>/<model>/valid/            train shard_019        (500 sims)
    <out_dir>/<model>/test/             test shard_000..004    (2,500 sims)
    <out_dir>/<model>/test_normalized/  the normalized test set (1,000 sims), to compare with the TCN AUCs

Each pickle mimics a NeuronIO file as read by `parse_sim_experiment_file`:

    Params:  allSegmentsType (placeholder, only its length N is used), totalSimDurationInSec,
             numSimulations, initialization_ms, + provenance (model, split, source shard, ...)
    Results: listOfSingleSimulationDicts, one dict per simulation with
             exInputSpikeTimes / inhInputSpikeTimes  {segment: uint16 array of ms}, all N segments
             outputSpikeTimes                        float array, ms + 0.5 (ELM bins with int(t - 0.5))
             somaVoltageLowRes                       float32 (ms,)

Simulations are kept whole, including the first SIMULATION_INITIALIZATION_MS (driven by noise
input), so a simulation is 10.5 s and totalSimDurationInSec = 10.5. ELM must skip the
initialization itself (ignore_time_from_start = initialization_ms) and convert the duration to an
int number of ms. --drop_initialization crops it instead (10 s files, initialization_ms = 0).
A synaptic weight w > 1 (w spikes in the same ms on one segment, <1% of events) is stored as
that time repeated w times. ELM's stock parser collapses repeats to one binary spike; a parser
that counts repeats recovers the weights exactly.

Usage (fci env, which has h5py):

    python utils/convert_shards_to_neuronio.py --model Rat_L2_TPC --list_jobs
    python utils/convert_shards_to_neuronio.py --model Rat_L2_TPC --max_files 1      # quick test
    python utils/convert_shards_to_neuronio.py --model Rat_L2_TPC --job_index $SLURM_ARRAY_TASK_ID
    python utils/convert_shards_to_neuronio.py --model Rat_L2_TPC                    # everything
"""

import os
import sys
import glob
import pickle
import argparse

import h5py
import numpy as np

# run as a script, this folder is sys.path[0], and its utils.py shadows the utils package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_shards import SHARD_GLOB, ShardIndex, decode_csr, read_simulation_group

DEFAULT_MODELS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'paper_data', 'simulations', 'models')
DEFAULT_OUT_DIR = os.path.join(os.environ.get('WORK', '/mnt/lustre/work/levina/bep519'), 'datasets', 'fci_converted')
SIMULATION_INITIALIZATION_MS = 500
SIMS_PER_FILE = 125
VALID_SHARDS = 1  # the last VALID_SHARDS train shards become the validation set
OUT_SPLITS = ('train', 'valid', 'test', 'test_normalized')


def resolve_model_folder(models_folder, model):
    """Accept the full model folder name or a unique prefix of it (e.g. 'Rat_L2_TPC')."""
    if os.path.isdir(os.path.join(models_folder, model)):
        return os.path.join(models_folder, model)
    matches = sorted(d for d in os.listdir(models_folder) if d.startswith(model))
    if len(matches) != 1:
        raise ValueError(f"model '{model}' matches {len(matches)} folders in {models_folder}: {matches}")
    return os.path.join(models_folder, matches[0])


def shard_simulation_indices(shard_path):
    with h5py.File(shard_path, 'r') as h:
        return [int(i) for i in h.attrs['simulation_indices']]


def chunks(values, size):
    if len(values) % size:
        # ELM samples simulations from range(sim_per_file), so every file must be full
        raise ValueError(f"{len(values)} simulations do not split into files of {size}")
    return [values[i:i + size] for i in range(0, len(values), size)]


def build_jobs(model_folder, sims_per_file):
    """One job per output file: (out_split, source_split, file_name, simulation_indices)."""
    jobs = []
    train_shards = sorted(glob.glob(os.path.join(model_folder, 'train', SHARD_GLOB)))
    test_shards = sorted(glob.glob(os.path.join(model_folder, 'test', SHARD_GLOB)))
    sharded_splits = [('train', 'train', train_shards[:-VALID_SHARDS]),
                      ('valid', 'train', train_shards[-VALID_SHARDS:]),
                      ('test', 'test', test_shards)]
    for out_split, source_split, shard_paths in sharded_splits:
        for shard_path in shard_paths:
            shard_name = os.path.splitext(os.path.basename(shard_path))[0]
            for part, indices in enumerate(chunks(shard_simulation_indices(shard_path), sims_per_file)):
                jobs.append((out_split, source_split, f'{out_split}_sim__{source_split}_{shard_name}_part{part}.p', indices))

    test_summary = pickle.load(open(os.path.join(model_folder, 'test', 'summary.pkl'), 'rb'))
    normalized = sorted(int(i) for i in test_summary['firing_rate_information']['normalized_test_set_by_index'])
    for part, indices in enumerate(chunks(normalized, sims_per_file)):
        jobs.append(('test_normalized', 'test', f'test_normalized_sim__part{part:02d}.p', indices))
    return jobs


def csr_to_spike_time_dict(group, tag, start_ms):
    """
    {row: uint16 array of spike times (ms, relative to start_ms)} for every row of a CSR group,
    with a time repeated w times where the weight (spike count) is w.
    """
    shape = tuple(int(x) for x in group.attrs[f'{tag}_shape'])
    matrix = decode_csr(group[f'{tag}_indptr'][:], group[f'{tag}_indices_delta'][:], group[f'{tag}_data'][:], shape)
    matrix.sort_indices()
    spike_times = {}
    for row in range(shape[0]):
        times = matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]]
        weights = matrix.data[matrix.indptr[row]:matrix.indptr[row + 1]].astype(np.int64)
        keep = times >= start_ms
        spike_times[row] = np.repeat(times[keep] - start_ms, weights[keep]).astype(np.uint16)
    return spike_times, shape


def convert_simulation(group, start_ms):
    exc, exc_shape = csr_to_spike_time_dict(group, 'exc', start_ms)
    inh, inh_shape = csr_to_spike_time_dict(group, 'inh', start_ms)
    if exc_shape != inh_shape:
        raise ValueError(f"{group.name}: exc {exc_shape} and inh {inh_shape} shapes differ, ELM needs n_exc == n_inh")

    simulation = read_simulation_group(group, what=('somatic_voltage', 'summary'))
    spike_times = np.asarray(simulation['summary']['output_spike_times'], dtype=np.int64)
    spike_times = spike_times[spike_times >= start_ms] - start_ms
    sim_dict = dict(
        exInputSpikeTimes=exc,
        inhInputSpikeTimes=inh,
        # ELM puts a spike at bin int(t - 0.5), so t + 0.5 lands it on bin t
        outputSpikeTimes=spike_times.astype(np.float64) + 0.5,
        somaVoltageLowRes=simulation['somatic_voltage'][start_ms:].astype(np.float32),
    )
    return sim_dict, exc_shape


def run_job(job, model_folder, out_folder, train_summary, drop_initialization=False, overwrite=False):
    out_split, source_split, file_name, simulation_indices = job
    out_path = os.path.join(out_folder, out_split, file_name)
    if os.path.exists(out_path) and not overwrite:
        print(f"exists, skipping: {out_path}")
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    start_ms = SIMULATION_INITIALIZATION_MS if drop_initialization else 0
    shard_index = ShardIndex(os.path.join(model_folder, source_split))
    sim_dicts, shapes = [], set()
    for index in simulation_indices:
        sim_dict, shape = convert_simulation(shard_index.group(index), start_ms)
        sim_dicts.append(sim_dict)
        shapes.add(shape)
    source_shards = sorted({os.path.basename(shard_index.simulation_to_shard[i]) for i in simulation_indices})
    shard_index.close()
    if len(shapes) != 1:
        raise ValueError(f"{file_name}: simulations have different input shapes {shapes}")
    num_segments, duration_ms = shapes.pop()

    duration_ms -= start_ms
    voltage_information = train_summary['average_somatic_voltage_information']
    params = dict(
        allSegmentsType=['dend'] * num_segments,                 # placeholder, the shards have no segment types
        allSegments_DistFromSoma=[np.nan] * num_segments,        # placeholder, read only with include_params
        totalSimDurationInSec=duration_ms / 1000,                # 10.5: ELM must int() the duration in ms
        numSimulations=len(sim_dicts),
        initialization_ms=SIMULATION_INITIALIZATION_MS - start_ms,  # still in the data, for ELM to ignore
        n_exc=num_segments,
        n_inh=num_segments,
        model=os.path.basename(model_folder),
        split=out_split,
        source_split=source_split,
        source_shards=source_shards,
        simulation_indices=list(simulation_indices),
        dropped_initialization_ms=start_ms,
        # from train/summary.pkl, as FCI's TCN training uses for v_offset
        average_somatic_voltage=float(voltage_information['average_somatic_voltage']),
        average_clipped_somatic_voltage=float(voltage_information['average_clipped_somatic_voltage']),
    )
    experiment = dict(Params=params, Results=dict(listOfSingleSimulationDicts=sim_dicts))

    tmp_path = f'{out_path}.{os.getpid()}.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(experiment, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, out_path)
    print(f"wrote {out_path} ({len(sim_dicts)} sims, {2 * num_segments} inputs, {duration_ms} ms)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='model folder name, or a unique prefix of it')
    parser.add_argument('--models_folder', default=DEFAULT_MODELS_FOLDER)
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR, help='files go to <out_dir>/<model>/<split>/')
    parser.add_argument('--sims_per_file', type=int, default=SIMS_PER_FILE)
    parser.add_argument('--splits', nargs='+', default=list(OUT_SPLITS), choices=OUT_SPLITS)
    parser.add_argument('--job_index', type=int, default=None, help='run only this job (e.g. $SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--max_files', type=int, default=None, help='run only the first n jobs, for a quick test')
    parser.add_argument('--list_jobs', action='store_true', help='print the jobs and exit')
    parser.add_argument('--drop_initialization', action='store_true',
                        help=f'crop the first {SIMULATION_INITIALIZATION_MS} ms instead of keeping them')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    model_folder = resolve_model_folder(args.models_folder, args.model)
    out_folder = os.path.join(args.out_dir, os.path.basename(model_folder))
    jobs = [job for job in build_jobs(model_folder, args.sims_per_file) if job[0] in args.splits]

    if args.list_jobs:
        for i, (out_split, source_split, file_name, indices) in enumerate(jobs):
            print(f"{i:4d}  {out_split:16s} {file_name}  ({len(indices)} sims from {source_split})")
        print(f"{len(jobs)} jobs -> {out_folder}")
        return

    if args.job_index is not None:
        jobs = [jobs[args.job_index]]
    elif args.max_files is not None:
        jobs = jobs[:args.max_files]

    train_summary = pickle.load(open(os.path.join(model_folder, 'train', 'summary.pkl'), 'rb'))
    for job in jobs:
        run_job(job, model_folder, out_folder, train_summary,
                drop_initialization=args.drop_initialization, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
