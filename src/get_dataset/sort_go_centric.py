#!/usr/bin/python3

import argparse
import sys
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gocentric', required=True)
    parser.add_argument('-o', '--outfile', required=True)

    return vars(parser.parse_args())


def sort_chunk(chunk):
    try:
        header = chunk[0].strip()
        print(f'sorting go {header} with {len(chunk)} lines', file=sys.stderr)
        lines_to_sort = chunk[1:]
        sorted_chunk = [header]
        noniea_lines = []
        uprkw_lines = []
        other_lines = []
        for line in lines_to_sort:
            if int(line.split('\t')[1]) == 0:
                noniea_lines.append(line)
            elif int(line.split('\t')[1]) == 1:
                uprkw_lines.append(line)
            else:
                other_lines.append(line)

        sorted_chunk.extend(noniea_lines)
        sorted_chunk.extend(uprkw_lines)
        sorted_chunk.extend(other_lines)

        noniea = len(noniea_lines)
        uprkw = len(uprkw_lines)
        tot = noniea + uprkw
        header += f'|tot={tot}'
        header += f'|manual={noniea}'
        header += f'|uprkw={uprkw}\n'
        sorted_chunk[0] = header

        return sorted_chunk
    except Exception as e:
        print(f'Error occurred while sorting chunk: {e}', file=sys.stderr)
        return []


def get_chunks(infile):
    with open(infile, 'r') as infile:
        current_header = None
        current_chunk = []

        for line in infile:
            if line.startswith('>'):
                if current_header is not None:
                    yield current_chunk
                    current_chunk = []
                current_chunk.append(line)
                current_header = line.strip()
            else:
                current_chunk.append(line)

        yield current_chunk


if __name__ == '__main__':
    args = get_args()
    go_centric = args['gocentric']
    out_file = args['outfile']

    print(f'Getting chunk from {go_centric} AND saving to {out_file}...', file=sys.stderr)
    with open(out_file, 'w') as out:
        for chunk in get_chunks(go_centric):
            sorted_chunk = sort_chunk(chunk)
            for line in sorted_chunk:
                out.write(line)

