import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import re

# --- Wczytywanie profilu ONNX i przygotowanie DataFrame ---
def load_events(path):
    with open(path, 'r') as f:
        raw = json.load(f)
    records = []
    for e in raw:
        if e.get('ph') == 'X' and 'ts' in e and 'dur' in e:
            records.append({
                'name': e.get('name',''),
                'ts': e['ts'],
                'dur': e['dur'],
                'op_name': e.get('args',{}).get('op_name','Unknown')
            })
    df = pd.DataFrame(records).sort_values('ts').reset_index(drop=True)
    df['start_ms'] = (df['ts'] - df['ts'].min()) / 1000.0
    df['dur_ms']   = df['dur'] / 1000.0
    # mapowanie bloków wg nazw
    def extract_block(name):
        if '/layer1/' in name: return 'ResNet Layer 1'
        if '/layer2/' in name: return 'ResNet Layer 2'
        if '/layer3/' in name: return 'ResNet Layer 3'
        if '/lstm1/' in name: return 'BiLSTM 1'
        if '/lstm2/' in name: return 'BiLSTM 2'
        return 'Other'
    df['block'] = df['name'].apply(extract_block)
    return df

# --- Rysowanie wykresu Gantt dla wybranych bloków ---
def plot_section(df, section_blocks, title, output_pdf):
    blocks = [b for b in section_blocks if b in df['block'].unique()]
    if not blocks:
        print(f"Brak bloków do narysowania: {title}")
        return

    cum_time = 0.0
    fig, axes = plt.subplots(len(blocks), 1, figsize=(12, 3*len(blocks)), sharex=False)
    if len(blocks) == 1:
        axes = [axes]

    for idx, (ax, blk) in enumerate(zip(axes, blocks)):
        sub = df[df['block'] == blk].sort_values('start_ms').reset_index(drop=True)
        offset = sub['start_ms'].min()
        sub['adj_start'] = sub['start_ms'] - offset + cum_time

        ys = np.arange(len(sub))
        cmap = plt.get_cmap('tab20', sub['op_name'].nunique())
        colors = {op: cmap(i) for i, op in enumerate(sub['op_name'].unique())}

        for i, row in sub.iterrows():
            ax.barh(ys[i], row['dur_ms'], left=row['adj_start'], height=0.6,
                    color=colors[row['op_name']], edgecolor='black')

        # czerwona linia startu
        block_start = sub['adj_start'].min()
        ax.axvline(block_start, color='red', linestyle='--', linewidth=1.5)

        # zaktualizuj czas skumulowany
        cum_time = sub['adj_start'].max() + sub['dur_ms'].max()

        # etykiety y
        ax.set_yticks(ys)
        ax.set_yticklabels([n.split('/')[-1] for n in sub['name']], fontsize=8)

        # żółta etykieta bloku naprzemiennie
        y_text = ys.mean() + ((len(sub)*0.3) if idx % 2 == 0 else -(len(sub)*0.3))
        ax.text(block_start, y_text, blk, rotation=90,
                va='center', ha='center', fontsize=9,
                bbox=dict(facecolor='yellow', edgecolor='black', alpha=0.8))

        ax.set_xlabel(' Time (ms)')

    fig.suptitle(title, fontsize=14)
    # legenda
    ops = df['op_name'].unique()
    cmap_all = plt.get_cmap('tab20', len(ops))
    patches = [mpatches.Patch(color=cmap_all(i), label=op) for i, op in enumerate(ops)]
    fig.legend(handles=patches, loc='upper right', title='Operation types')

    plt.tight_layout(rect=[0,0,0.85,1])
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Zapisano: {output_pdf}")

# --- Główna część skryptu ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gantt ONNX profiling by section')
    parser.add_argument('-i', '--input',     required=True, help='Profiling JSON file')
    parser.add_argument('-o', '--out_prefix', default='gantt', help='Prefix for output PDFs')
    args = parser.parse_args()

    df = load_events(args.input)

    plot_section(df,
                 ['ResNet Layer 1','ResNet Layer 2','ResNet Layer 3'],
                 'ResNet Blocks ',
                 f"{args.out_prefix}_resnet.pdf")

    plot_section(df,
                 ['BiLSTM 1','BiLSTM 2'],
                 'BiLSTM Blocks ',
                 f"{args.out_prefix}_bilstm.pdf")
