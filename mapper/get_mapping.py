from get_mapper import *
from model import Aligner
import torch
import argparse



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--iter_norm", action="store_true")
    parser.add_argument("--anisotropy", action="store_true")

    args = parser.parse_args()

    device = torch.device("cpu")
    aligner = Aligner(768)

    if_csls = False
    flag = args.iter_norm


    trainer = Mapper(args.input, aligner, flag, device)

    if args.anisotropy:
        # index = random_indice_generator(1000, trainer.source_vector.shape[1])
        an_s = degree_anisotropy(trainer.source_vector)
        an_t = degree_anisotropy(trainer.target_vector)
        print("anisotriopy for source lang is {}, for taget lang is {} ".format(float(an_s), float(an_t)))
        return

    W, aligned = trainer.simple_procrustes()

    torch.save(W, "/export/b15/haoranxu/clce/mappings/no_noise/"+ args.prefix + args.input.split("/")[-1] + ".th")

if __name__ == '__main__':
    main()