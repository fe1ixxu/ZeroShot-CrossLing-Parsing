{
    "dataset_reader": {
        "type": "universal_dependencies",
    },
    "vocabulary": {
        type:"from_files",
        directory: "/export/b15/haoranxu/clce/fasttext/vocabulary/",
        oov_token: "@@UNKNOWN@@"
    },
    "model": {
        "type": "biaffine_parser",
        "arc_representation_dim": 500,
        "dropout": 0.3,
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "hidden_size": 500,
            "input_size": 400,
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
                    "type": "embedding",
                    "embedding_dim": 300,
                    "pretrained_file": "/export/b15/haoranxu/clce/fasttext/wiki.en.align.vec",
                    "sparse": true,
                    "trainable": false
                }
            }
        },
        "use_mst_decoding_for_validation": true
    },
    "train_data_path": "/export/b15/haoranxu/clce/data/ud-treebanks-v2.6/UD_English-EWT/en_ewt-ud-train.conllu",  
    "validation_data_path": "/export/b15/haoranxu/clce/data/ud-treebanks-v2.6/UD_English-EWT/en_ewt-ud-dev.conllu",
    "trainer": {
        "cuda_device": 0,
        "grad_norm": 5,
        "num_epochs": 200,
        "optimizer": {
            "type": "dense_sparse_adam",
            "betas": [
                0.9,
                0.9
            ]
        },
        "patience": 25,
        "validation_metric": "+LAS"
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 128
        }
    }
}