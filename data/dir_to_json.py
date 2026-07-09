import os
import re
import json
import argparse


# Matches the SNR suffix EARS-WHAM appends to noisy filenames, e.g.
# "00002_-2.2dB.wav" -> snr = -2.2. Group 1 is the numeric dB value.
SNR_RE = re.compile(r'_(-?\d+(?:\.\d+)?)dB\.wav$', re.IGNORECASE)


def parse_snr(filename):
    """Return (stripped_name, snr_float) or (None, None) if not annotated."""
    m = SNR_RE.search(filename)
    if m is None:
        return None, None
    stripped = filename[:m.start()] + '.wav'
    return stripped, float(m.group(1))


def list_wav_files(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for name in filenames:
            if name.lower().endswith('.wav'):
                files.append(os.path.join(root, name))
    files.sort()
    return files


def write_json(files, output_file):
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(files, f, indent=4)
    print(f'Wrote {len(files)} paths to {output_file}')


# EARS-WHAM / VB-DMD share this layout:
#   <root>/<split>/<clean|noisy>/<speaker>/*.wav
# Manifests are named <manifest_split>_<clean|noisy>.json.
SPLITS = {'train': 'train', 'valid': 'valid', 'test': 'test'}
# Accept common spellings for the clean subdir ("clear" as the user calls it).
CLEAN_ALIASES = ('clean', 'clear')
NOISY_ALIASES = ('noisy',)


def resolve_subdir(split_dir, aliases):
    for name in aliases:
        candidate = os.path.join(split_dir, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def write_snr_manifest(files, root, output_file):
    """Map each noisy file's annotation-stripped path (relative to root) to its
    SNR in dB, so datasets without the annotation can be matched by that key."""
    snr = {}
    unannotated = 0
    for path in files:
        stripped_name, value = parse_snr(os.path.basename(path))
        if value is None:
            unannotated += 1
            continue
        rel = os.path.relpath(path, root)
        key = os.path.join(os.path.dirname(rel), stripped_name)
        snr[key] = value
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(snr, f, indent=4)
    msg = f'Wrote {len(snr)} SNR entries to {output_file}'
    if unannotated:
        msg += f' ({unannotated} files had no dB annotation, skipped)'
    print(msg)


def build_dataset(root, output_dir, snr=True):
    for split_name, manifest_split in SPLITS.items():
        split_dir = os.path.join(root, split_name)
        if not os.path.isdir(split_dir):
            print(f'Skipping missing split: {split_dir}')
            continue
        for kind, aliases in (('clean', CLEAN_ALIASES), ('noisy', NOISY_ALIASES)):
            subdir = resolve_subdir(split_dir, aliases)
            if subdir is None:
                print(f'Skipping missing {kind} dir under {split_dir}')
                continue
            files = list_wav_files(subdir)
            out = os.path.join(output_dir, f'{manifest_split}_{kind}.json')
            write_json(files, out)
            if snr and kind == 'noisy':
                snr_out = os.path.join(output_dir, f'{manifest_split}_snr.json')
                write_snr_manifest(files, root, snr_out)


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Write JSON lists of .wav paths. Either scan a single directory, '
            'or scan an EARS-WHAM/VB-DMD dataset root '
            '(<root>/{train,valid,test}/{clean,noisy}/<speaker>/*.wav) and emit '
            'all six manifests.'
        )
    )
    parser.add_argument(
        '--dataset_root',
        help='Dataset root with train/valid/test splits. Emits all six manifests '
             'into --output_dir.',
    )
    parser.add_argument(
        '--output_dir',
        help='Where to write the six manifests (dataset-root mode). '
             'Defaults to data/ears_wham_16k.',
    )
    parser.add_argument(
        '--no_snr',
        action='store_true',
        help='Skip writing per-split SNR manifests (dataset-root mode). By '
             'default an <split>_snr.json mapping noisy paths to their dB SNR '
             'is emitted alongside the noisy manifest.',
    )
    parser.add_argument(
        'input_dir',
        nargs='?',
        help='Single directory to scan recursively (legacy mode).',
    )
    parser.add_argument(
        'output_file',
        nargs='?',
        help='Output JSON path (legacy single-directory mode).',
    )
    args = parser.parse_args()

    if args.dataset_root:
        output_dir = args.output_dir or os.path.join('data', 'ears_wham_16k')
        build_dataset(args.dataset_root, output_dir, snr=not args.no_snr)
        return

    if not (args.input_dir and args.output_file):
        parser.error(
            'Provide either --dataset_root, or both input_dir and output_file.'
        )

    files = list_wav_files(args.input_dir)
    write_json(files, args.output_file)


if __name__ == '__main__':
    main()
