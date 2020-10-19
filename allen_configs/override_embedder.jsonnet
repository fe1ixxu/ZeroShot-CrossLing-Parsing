local overrides_model = "override_place_holder";
local map_path = "map_path_place_holder";


{ 
    "new_pretrained_embedder": overrides_model,
        "dataset_reader": {
            "token_indexers": {
                "tokens": {
                "model_name": overrides_model
                }
            }
        },
    "model": {
        "new_pretrained_embedder": overrides_model,
        "text_field_embedder": { 
            "token_embedders": { 
                "tokens": { 
                    "model_name": overrides_model, 
                    "map_path": map_path
                }
            }
        }
    }
}