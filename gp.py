"""Run G0, G1 and G2 on prepared SMOTE-ENN training windows."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import numpy as np
import pandas as pd
from clean import cli_config, load_state, save_state


def compile_java(output):
    compiler = shutil.which('javac')
    if compiler is None:
        raise RuntimeError('Install JDK 17 and put javac and java on PATH.')
    build = Path(output) / 'work' / 'java'
    build.mkdir(parents=True, exist_ok=True)
    sources = [str(Path(__file__).with_name(v + '.java')) for v in ('G0', 'G1', 'G2')]
    result = subprocess.run([compiler, '--release', '17', '-encoding', 'UTF-8', '-d', str(build), *sources],
                            capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Java compilation failed:\n' + result.stderr)
    return build


def run_window(window, variant, cfg, build):
    x, y = np.asarray(window['x_train']), np.asarray(window['y_train'])
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError('GP needs finite training features and both target classes.')
    java = shutil.which('java')
    if java is None:
        raise RuntimeError('Install JDK 17 and put java on PATH.')
    index = int(window['index'])
    seed = int(cfg['seed']) + index
    flags = {'--population': 'population', '--generations': 'generations', '--depth': 'initial_depth',
             '--tournament': 'tournament_size', '--crossover': 'crossover_probability',
             '--max-length': 'max_tree_length', '--lambda': 'parsimony_lambda',
             '--tree-norm': 'tree_size_normalizer', '--archive-budget': 'archive_size'}
    with tempfile.TemporaryDirectory(prefix=f'{variant}_{index}_', dir=build.parent) as scratch:
        temp = Path(scratch).resolve()
        if not temp.is_relative_to(build.parent.resolve()):
            raise ValueError('Temporary output is outside the working directory.')
        train, archive = temp / 'train.txt', temp / 'candidates.tsv'
        np.savetxt(train, np.column_stack([x, y]), fmt='%.17g',
                   header=f'{x.shape[1]} 0 -1 1 {len(y)}', comments='')
        command = [java, '-cp', str(build), variant, '--data', str(train), '--candidates', str(archive),
                   '--seed', str(seed), '--window', str(index), '--replicate', '1']
        for flag, key in flags.items():
            command.extend([flag, str(cfg['gp'][key])])
        start = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        seconds = time.perf_counter() - start
        if result.returncode:
            raise RuntimeError(f'{variant}, window {index}:\n{result.stderr[-3000:]}')
        champions = re.findall(r'^Best Individual: (.+)$', result.stdout, flags=re.MULTILINE)
        if len(champions) != int(cfg['gp']['generations']):
            raise RuntimeError('GP generation count differs from the configured count.')
        frame = pd.read_csv(archive, sep='\t')
    if frame.empty or len(frame) > int(cfg['gp']['archive_size']):
        raise RuntimeError('GP archive size is outside the configured bounds.')
    raw_count = int(frame['n_raw_archive_observations'].iloc[0]) if 'n_raw_archive_observations' in frame else len(frame)
    frame = frame.rename(columns={'expression_exact': 'expression', 'window': 'source_window'})
    frame['variant'], frame['source_window'] = variant, index
    frame['candidate_id'] = [f'{variant}_w{index}_c{i}' for i in range(1, len(frame) + 1)]
    columns = ['candidate_id', 'variant', 'source_window', 'generation', 'expression']
    columns += [c for c in ('evolution_auc', 'training_auc', 'tree_size') if c in frame]
    champion = {'candidate_id': f'{variant}_w{index}_gp', 'variant': variant, 'source_window': index,
                'expression': champions[-1].strip(), 'generation': int(cfg['gp']['generations']) - 1}
    summary = {'variant': variant, 'source_window': index, 'training_rows': len(y),
               'candidate_observations': raw_count, 'archive_candidates': len(frame), 'seconds': seconds}
    return frame[columns], champion, summary


def main():
    cfg = cli_config(__doc__)
    state = load_state(cfg)
    if not state.get('features') or any('x_train' not in w for w in state.get('windows', [])):
        raise ValueError('Run features.py before gp.py.')
    build = compile_java(cfg['output'])
    candidates, champions, timing = [], [], []
    for variant in cfg['gp']['variants']:
        for window in state['windows']:
            print(f'{variant}: window {window["index"]}', flush=True)
            archive, champion, summary = run_window(window, variant, cfg, build)
            candidates.append(archive)
            champions.append(champion)
            timing.append(summary)
    for key in list(state):
        if key not in {'data', 'candidate_features', 'features', 'windows', 'feature_ranking', 'feature_curve', 'config'}:
            del state[key]
    state['candidates'] = pd.concat(candidates, ignore_index=True)
    state['gp_only'] = pd.DataFrame(champions)
    state['gp_timing'] = pd.DataFrame(timing)
    state['candidates'].to_csv(Path(cfg['output']) / 'candidates.csv', index=False)
    save_state(cfg, state)
    print(f'Saved {len(state["candidates"])} candidates to {cfg["output"]}')


if __name__ == '__main__':
    main()

