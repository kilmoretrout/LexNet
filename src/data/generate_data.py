import os
import numpy as np
import logging, argparse

import configparser

def parse_args():
    # Argument Parser    
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--odir", default = "/proj/dschridelab/introgression_data/")
    parser.add_argument("--n_jobs", default = "1000")
    parser.add_argument("--n_replicates", default = "10000")

    # simulation SLiM script
    parser.add_argument("--slim_file", default = "src/SLiM/introg_bidirectional.slim")

    # parameters for the simulation
    parser.add_argument("--st", default = "4")
    parser.add_argument("--mt", default = "0.25")
    parser.add_argument("--mp", default = "1")
    parser.add_argument("--phys_len", default = "3000")
    parser.add_argument("--donor_pop", default="1")

    parser.add_argument("--n_per_pop", default = "64")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.odir):
        os.mkdir(args.odir)
        logging.debug('root: made output directory {0}'.format(args.odir))
    else:
        os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    # config file with the population size in it (maybe other stuff)?
    config = configparser.ConfigParser()
    config['info'] = {'n_per_pop': args.n_per_pop, 'n_per_file': str(int(args.n_replicates) // int(args.n_jobs))}

    with open(os.path.join(args.odir, 'info.config'), 'w') as configfile:
        config.write(configfile)

    return args

def main():
    args = parse_args()

    n_jobs = int(args.n_jobs)
    n_replicates = int(args.n_replicates)

    replicates_per = n_replicates // n_jobs

    # should have an even number of pop1 to pop2 vs. pop2 to pop1
    # eventually we'll put in introgression both ways
    cmd = 'sbatch -o {0} --array=1-{1} src/SLURM/generate_data.sh {2} {3} {4} {5} {6} {7} {8} {9} {10}'.format(os.path.join(args.odir, 'sim.%a.out'), n_jobs, args.st, args.mt, args.mp, args.phys_len, args.n_per_pop, replicates_per, args.donor_pop, args.odir, args.slim_file)
    print(cmd)

    job_id = os.popen(cmd).read().split(" ")[-1].strip('\n')
    print('array jobid is {}'.format(job_id))

    gzip_cmd = 'sbatch -t 1-00:00:00 --depend=afterok:{0} --wrap "gzip {1} {2}"'.format(job_id, os.path.join(args.odir, '*.ms'), os.path.join(args.odir, '*.log'))
    os.system(gzip_cmd)


if __name__ == '__main__':
    main()
    
    
