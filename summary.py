"""Summarize GP candidate counts and elapsed time by window."""
from pathlib import Path
from clean import cli_config, load_state


def main():
    cfg = cli_config(__doc__)
    state = load_state(cfg)
    table = state['gp_timing'].copy()
    for key, column in [('menus', 'window_selected'), ('selected', 'global_selected')]:
        if key not in state:
            raise ValueError('Run core.py before summary.py.')
        counts = state[key].groupby(['variant', 'source_window']).size().rename(column).reset_index()
        table = table.merge(counts, on=['variant', 'source_window'], how='left')
        table[column] = table[column].fillna(0).astype(int)
    table.to_csv(Path(cfg['output']) / 'gp.csv', index=False)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()

