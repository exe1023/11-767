import argparse
import ipdb
import logging
import os
import sys


def main(args):
    watts = []
    for path in args.paths:
        watt_all = 0
        with open(path) as f:
            for line in f:
                if 'ipykernel_launcher' in line and '%' in line:
                    watt = line.split(';')[-1].strip()
                    if 'mW' in watt:
                        watt = float(watt.replace('mW', '').strip())
                        watt_all += watt / 1000
                    else:
                        watt = float(watt.replace('W', '').strip())
                        watt_all += watt
        watts.append(watt_all)

    print(f'Avg watt = {sum(watts) / len(watts)}')


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('paths', type=str, nargs='+', help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        main(args)
