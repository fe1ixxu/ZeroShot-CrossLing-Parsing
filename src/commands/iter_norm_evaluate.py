"""
The `evaluate` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.
"""

import argparse
import json
import logging
from typing import Any, Dict
import torch

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import dump_metrics, prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import DataLoader
from allennlp.models.archival import load_archive
from allennlp.training.util import evaluate
from allennlp.models.model import Model
from allennlp.common import Lazy, Registrable, Tqdm
from allennlp.common.checks import check_for_gpu
from allennlp.nn import util as nn_util

logger = logging.getLogger(__name__)


@Subcommand.register("iter-norm-evaluate")
class Iter_Norm_Evaluate(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Evaluate the specified model + dataset"""
        subparser = parser.add_parser(
            self.name, description=description, help="Evaluate the specified model + dataset."
        )

        subparser.add_argument("archive_file", type=str, help="path to an archived trained model")

        subparser.add_argument(
            "input_file", type=str, help="path to the file containing the evaluation data"
        )

        subparser.add_argument("--output-file", type=str, help="path to output file")

        subparser.add_argument(
            "--weights-file", type=str, help="a path that overrides which weights file to use"
        )

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help="a JSON structure used to override the experiment configuration",
        )

        subparser.add_argument(
            "--batch-size", type=int, help="If non-empty, the batch size to use during evaluation."
        )

        subparser.add_argument(
            "--batch-weight-key",
            type=str,
            default="",
            help="If non-empty, name of metric used to weight the loss on a per-batch basis.",
        )

        subparser.add_argument(
            "--extend-vocab",
            action="store_true",
            default=False,
            help="if specified, we will use the instances in your new dataset to "
            "extend your vocabulary. If pretrained-file was used to initialize "
            "embedding layers, you may also need to pass --embedding-sources-mapping.",
        )

        subparser.add_argument(
            "--embedding-sources-mapping",
            type=str,
            default="",
            help="a JSON dict defining mapping from embedding module path to embedding "
            "pretrained-file used during training. If not passed, and embedding needs to be "
            "extended, we will try to use the original file paths used during training. If "
            "they are not available we will use random vectors for embedding extension.",
        )

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(
        args.archive_file,
        weights_file=args.weights_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.pop("validation_dataset_reader", None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop("dataset_reader"))
    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    embedding_sources = (
        json.loads(args.embedding_sources_mapping) if args.embedding_sources_mapping else {}
    )

    if args.extend_vocab:
        logger.info("Vocabulary is being extended with test instances.")
        model.vocab.extend_from_instances(instances=instances)
        model.extend_embedder_vocab(embedding_sources)

    instances.index_with(model.vocab)
    data_loader_params = config.pop("validation_data_loader", None)
    if data_loader_params is None:
        data_loader_params = config.pop("data_loader")
    if args.batch_size:
        data_loader_params["batch_size"] = args.batch_size
    data_loader = DataLoader.from_params(dataset=instances, params=data_loader_params)

    iter_num = model.text_field_embedder._token_embedders['tokens'].iter_norm
    if iter_num:
        # Obtrain evaluation info for iterative normalization:
        iter_mean_eval = []
        mean = torch.tensor([0.], device=args.cuda_device) #Initialize the embedding mean as 0
        for iter_norm_i in range(iter_num):
            logging.info("This is the {} time during iterative normalization for evaluation".format(iter_norm_i))
            mean = get_iter_norm_mean_eval(model, data_loader, mean, args.cuda_device)
            iter_mean_eval.append(mean)

        model.text_field_embedder._token_embedders['tokens'].iter_norm = None 
        model.text_field_embedder._token_embedders['tokens']._matched_embedder.mean_emb_eval = iter_mean_eval
        model.text_field_embedder._token_embedders['tokens']._matched_embedder.is_train = False

    metrics = evaluate(model, data_loader, args.cuda_device, args.batch_weight_key)

    logger.info("Finished evaluating.")

    dump_metrics(args.output_file, metrics, log=True)

    return metrics

def get_iter_norm_mean_eval(
    model: Model, 
    data_loader: DataLoader, 
    mean: torch.Tensor,
    cuda_device: int = -1
) -> Dict[str, Any]:
    """
    # Parameters

    model : `Model`
        The model to evaluate
    data_loader : `DataLoader`
        The `DataLoader` that will iterate over the evaluation data (data loaders already contain
        their data).
    cuda_device : `int`, optional (default=`-1`)
        The cuda device to use for this evaluation.  The model is assumed to already be using this
        device; this parameter is only used for moving the input data to the correct device.
    batch_weight_key : `str`, optional (default=`None`)
        If given, this is a key in the output dictionary for each batch that specifies how to weight
        the loss for that batch.  If this is not given, we use a weight of 1 for every batch.
    """
    check_for_gpu(cuda_device)
    with torch.no_grad():
        model.eval()

        iterator = iter(data_loader)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator)

        mean_embeddings: [torch.Tensor, int]
        mean_embeddings = [torch.tensor([0.], device=cuda_device), 0]
        for batch in generator_tqdm:
            batch = nn_util.move_to_device(batch, cuda_device)
            batch_embeddings = model.forward_embeddings(batch['words'], mean)
            mean_embeddings[0] = (mean_embeddings[0] + batch_embeddings.sum(dim=0)) / (mean_embeddings[1] + batch_embeddings.shape[0])
            mean_embeddings[1] += batch_embeddings.shape[0]

    return mean_embeddings[0]
