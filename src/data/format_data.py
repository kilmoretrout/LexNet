import os
import numpy as np
import logging, argparse

from data_functions import load_data_dros, get_params, sort_XY, get_windows_snps, two_channel_transform

import h5py

import configparser
from mpi4py import MPI
import sys

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--ofile", default = "None")

    parser.add_argument("--chunk_size", default = "12")
    parser.add_argument("--zero_check", action = "store_true")

    parser.add_argument("--max_snps", default = "None")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    # configure MPI
    comm = MPI.COMM_WORLD

    ifiles = [os.path.join(args.idir, u) for u in os.listdir(args.idir)]
    n_files = len(ifiles)

    chunk_size = int(args.chunk_size)
    max_snps = int(args.max_snps)

    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(ifiles), comm.size - 1):
            ifile = ifiles[ix]
            ifile = np.load(ifile)
            x_ = ifile['x']

            # destroy the perfect information regarding
            # which allele is the ancestral one
            for k in range(x_.shape[1]):
                if np.sum(x_[:, k]) > 32:
                    x_[:, k] = 1 - x_[:, k]
                elif np.sum(x_[:, k]) == 32:
                    if np.random.choice([0, 1]) == 0:
                        x_[:, k] = 1 - x_[:, k]

            y_ = ifile['y']

            if args.zero_check:
                if np.sum(y_) == 0:
                    comm.send([], dest = 0)
                    continue
            
            x = np.zeros((max_snps, x_.shape[0]), dtype = np.uint8)
            x[:x_.shape[1],:] = x_.T

            X = [x]

            comm.send(X, dest = 0)

    else:
        ofile = h5py.File(args.ofile, 'w')

        n_received = 0
        current_chunk = 0

        X = []

        while n_received < n_files:
            x = comm.recv(source = MPI.ANY_SOURCE)

            X.extend(x)

            n_received += 1

            while len(X) >= chunk_size:
                ofile.create_dataset('{0}/x_0'.format(current_chunk), data = np.array(X[-chunk_size:], dtype = np.uint8), compression = 'lzf')

                del X[-chunk_size:]

                current_chunk += 1

            ofile.flush()

        ofile.close()

if __name__ == '__main__':
    main()

"""
sbatch -n 24 --mem=64g -t 2-00:00:00 --wrap "mpirun python3 src/data/format_data_Lex.py --verbose --idir /proj/dschridelab/introgression_data/sims_drosophila/npzs_train/mig12_npzs/ --ofile /proj/dschridelab/ddray/dros_AB_10e6_rf_Lex.hdf5 --max_snps 1280"
sbatch -n 24 --mem=64g -t 2-00:00:00 --wrap "mpirun python3 src/data/format_data_Lex.py --verbose --idir /proj/dschridelab/introgression_data/sims_drosophila/npzs_train/mig21_npzs/ --ofile /proj/dschridelab/ddray/dros_BA_10e6_rf_Lex.hdf5 --max_snps 1280"
sbatch -n 24 --mem=64g -t 2-00:00:00 --wrap "mpirun python3 src/data/format_data_Lex.py --verbose --idir /proj/dschridelab/introgression_data/sims_drosophila/npzs_train/noMig_npzs/ --ofile /proj/dschridelab/ddray/dros_none_10e6_rf_Lex.hdf5 --max_snps 1280"
"""
