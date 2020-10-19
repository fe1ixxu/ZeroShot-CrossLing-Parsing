local bert_model = "bert-base-multilingual-uncased";
local max_length = 1024;
{
    "dataset_reader": {
        "type": "universal_dependencies",
        "token_indexers": {
          "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": bert_model,
            "max_length": max_length
          },
        },
    },
    "vocabulary": {
        "tokens_to_add": {
          "pos": ["@@UNKNOWN@@"],
          "head_tags": ["@@UNKNOWN@@"]
        }
    },
    "model": {
        "type": "bert_biaffine_parser",
        "arc_representation_dim": 500,
        "dropout": 0.3,
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "hidden_size": 500,
            "input_size": 868,
            "num_layers": 3,
            "recurrent_dropout_probability": 0.3,
            "use_highway": true
        },
        "initializer": {
            "regexes": [
                [
                    ".*projection.*weight",
                    {
                        "type": "xavier_uniform"
                    }
                ],
                [
                    ".*projection.*bias",
                    {
                        "type": "zero"
                    }
                ],
                [
                    ".*tag_bilinear.*weight",
                    {
                        "type": "xavier_uniform"
                    }
                ],
                [
                    ".*tag_bilinear.*bias",
                    {
                        "type": "zero"
                    }
                ],
                [
                    ".*weight_ih.*",
                    {
                        "type": "xavier_uniform"
                    }
                ],
                [
                    ".*weight_hh.*",
                    {
                        "type": "orthogonal"
                    }
                ],
                [
                    ".*bias_ih.*",
                    {
                        "type": "zero"
                    }
                ],
                [
                    ".*bias_hh.*",
                    {
                        "type": "lstm_hidden_bias"
                    }
                ]
            ]
        },
        "input_dropout": 0.3,
        "pos_tag_embedding": {
            "embedding_dim": 100,
            "sparse": true,
            "vocab_namespace": "pos"
        },
        "tag_representation_dim": 100,
        "text_field_embedder": {
          "token_embedders": {
            "tokens": {
                "type": "static_pretrained_transformer_mismatched",
                "model_name": bert_model,
                "max_length": max_length,
                "train_parameters": false,
                "if_top_layers": true,
                "if_normalize": true,
                "iter_norm": 5
            }
          }
        },
        "use_mst_decoding_for_validation": true
    },
    "train_data_path": "/export/b15/haoranxu/clce/data/ud-treebanks-v2.6/UD_English-EWT/en_ewt-ud-train.conllu",  
    "validation_data_path": "/export/b15/haoranxu/clce/data/ud-treebanks-v2.6/UD_English-EWT/en_ewt-ud-dev.conllu",
    "trainer": {
        "type": "iter_norm",
        "cuda_device": 0,
        "grad_norm": 5,
        "num_epochs": 200,
        "optimizer": {
            "lr": 0.0008,
            "type": "dense_sparse_adam",
            "betas": [
                0.9,
                0.9
            ]
        },
        "patience": 10,
        "validation_metric": "+LAS"
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 128
        }
    }
}
