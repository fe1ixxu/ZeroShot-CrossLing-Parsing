from get_mapper import *
import torch
import argparse



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--iter_norm", action="store_true")
    parser.add_argument("--anisotropy", action="store_true")
    
    args = parser.parse_args()
    flag = args.iter_norm

    trainer = Mapper(args.input, flag)

    if args.anisotropy:
        an_s = degree_anisotropy(trainer.source_vector)
        an_t = degree_anisotropy(trainer.target_vector)
        print("anisotriopy for source lang is {}, for taget lang is {} ".format(float(an_s), float(an_t)))
        return

    W, aligned = trainer.simple_procrustes()

    torch.save(W, args.output + args.input.split("/")[-1] + ".th")

if __name__ == '__main__':
    main()