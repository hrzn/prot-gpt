# prot-gpt
Nano prot gpt


## Procedure
1. Prepare Python
```
$ pip install -r requirements.txt
```

2. Download sequences from PDB:
```
$ mkdir data && cd data
$ wget https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz
$ gzip -d pdb_seqres.txt.gz && cd ..
```

3. Pre-process sequences:
```
$ python preprocess_pdb_seqres.py
```
That creates a file `data/prot_seqs.txt`, which contains `mol:protein` entries of the PDB file (one entry per distinct name).

4. Choose hyper-parameters in `train_proteins.py` and train model:
```
$ python train_proteins.py
```
You can launch a Tensorboard instance to watch the model being trained.

At the end (or if CTRL+C'ing) the path to the best model checkpoint should be displayed.

5. Generate 100 proteins using a checkpointed model:
```
$ python generate_proteins.py 100 path/to/checkpoint.ckpt
```
This writes the generated proteins in a file `generated_proteins.txt`.

6. Visualise with AlphaFold:
Use the [AlphaFold Colab](https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb) with your own sequences!