import argparse
import json
import os
try:
    from _jsonnet import evaluate_file, evaluate_snippet
except ImportError:

    def evaluate_file(filename: str, **_kwargs) -> str:
        logger.warning(
            f"error loading _jsonnet (this is expected on Windows), treating {filename} as plain json"
        )
        with open(filename, "r") as evaluation_file:
            return evaluation_file.read()

    def evaluate_snippet(_filename: str, expr: str, **_kwargs) -> str:
        logger.warning(
            "error loading _jsonnet (this is expected on Windows), treating snippet as plain json"
        )
        return expr

def write_override_file(args, overrides):
    overrides = overrides.replace(" ", "")
    overrides = overrides.replace("\n", "")
    overrides = overrides.replace("\t", "")
    if args.lang == "fi":
        overrides = overrides.replace("override_place_holder", "TurkuNLP/bert-base-finnish-uncased-v1")
        overrides = overrides.replace("map_path_place_holder", args.mapping_path + "/20-iter-norm-" + args.type + "_fi-en-1m.th")
    elif args.lang == "el":
        overrides = overrides.replace("override_place_holder", "nlpaueb/bert-base-greek-uncased-v1")
        overrides = overrides.replace("map_path_place_holder", args.mapping_path + "/20-iter-norm-" + args.type + "_el-en-1m.th")
    elif args.lang == "es":
        overrides = overrides.replace("override_place_holder", "dccuchile/bert-base-spanish-wwm-uncased")
        overrides = overrides.replace("map_path_place_holder", args.mapping_path + "/20-iter-norm-" + args.type + "_es-en-1m.th")
    elif args.lang == "pl":
        overrides = overrides.replace("override_place_holder", "dkleczek/bert-base-polish-uncased-v1")
        overrides = overrides.replace("map_path_place_holder", args.mapping_path + "/15-iter-norm-" + args.type + "_pl-en-1m.th")
    elif args.lang == "ro":
        overrides = overrides.replace("override_place_holder", "dumitrescustefan/bert-base-romanian-uncased-v1")
        overrides = overrides.replace("map_path_place_holder", args.mapping_path + "/20-iter-norm-" + args.type + "_ro-en-1m.th")
    elif args.lang == "pt":
        overrides = overrides.replace("override_place_holder", "neuralmind/bert-base-portuguese-cased")
        overrides = overrides.replace("map_path_place_holder", args.mapping_path + "/20-iter-norm-" + args.type + "_pt-en-1m.th")

    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overrides", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--mapping_path", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--ifstatic", type=str, default=False)
    args = parser.parse_args()

    if args.ifstatic == "yes":
        overrides="{'model':{'text_field_embedder':{'token_embedders':{'tokens':{'pretrained_file':'override_place_holder'}}}}}"
        override_file =  "/export/b15/haoranxu/clce/fasttext/wiki." + args.lang + ".align.vec" 
        overrides = overrides.replace("override_place_holder", override_file)
    else:
        overrides = evaluate_file(args.overrides)
        overrides = write_override_file(args, overrides)

    print(overrides)

if __name__ == "__main__":
    main()


