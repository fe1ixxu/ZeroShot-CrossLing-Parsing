# Insturction for Deriving Cross-Lingual Space Mapping
### 1. Installing Fast Align

Please install the [fast align](https://github.com/clab/fast_align) toolkit by following their instructions.

### 2. Download and Preprocess Parallel Corpora

Parallel Corpora are downloaded from [ParaCrawl](https://www.paracrawl.eu/). A preprocessing file is already prepared for you and you can easily run it by one command. An example of downloading and preprocessing English-Finnish parallel corpora:
```
wget https://s3.amazonaws.com/web-language-models/paracrawl/release6/en-fi.txt.gz
gunzip ./en-fi.txt.gz
./data/preprocess.sh fi YOUR/PATH/FOR/PARALLEL/CORPUS YOUR/PATH/FOR/FAST/ALIGN
```

### 3. Obtain Contextual Embeddings

Continuing the above example, we run `getwordvectorsfrombert.py` to obtain aligned contextual embeddings.
```
path=YOUR/PATH/FOR/PARALLEL/CORPUS
python getwordvectorsfrombert.py --src fi --tgt en --open_src_file ${path}fi_token.txt  --open_tgt_file ${path}en_token.txt  --open_align_file ${path}forward_align.txt --write_vectors_path ${path}vectors/ --max_num_word 10000 --batch_size 256 --max_seq_length 150
```
### 4. Cluster aligned Embeddings

To obtain sense-level embeddings:
```
OUTPUT=YOUR/OUTPUT/PATH
python cluster_vector.py --input_file ${path}vectors/ --write_file $OUTPUT  --min_threshold 100 --min_num_words 5 
```

### 5. Derive Linear Mapping
```
PATH_MAPPING=PATH/TO/STORE/MAPPIG
python get_mapping.py --input $OUTPUT --output $PATH_MAPPING --iter_norm 
```
If you want to see the degree of anisotropy for the source and the target languages, please run:

before iterative normalization:
```
python get_mapping.py --input $OUTPUT --anisotropy
```
after iterative normalization:
```
python get_mapping.py --input $OUTPUT --anisotropy --iter_norm 
```
