import os
import json
import argparse


def list_wav_files(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for name in filenames:
            if name.lower().endswith('.wav'):
                files.append(os.path.join(root, name))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Write a JSON list of .wav file paths in a directory.'
    )
    parser.add_argument(
        'input_dir',
        help='Directory to scan for .wav files (recursive).'
    )
    parser.add_argument(
        'output_file',
        help='Path to the output JSON file.'
    )
    args = parser.parse_args()

    files = list_wav_files(args.input_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(files, f, indent=4)
    print(f'Wrote {len(files)} paths to {args.output_file}')


if __name__ == '__main__':
    main()
