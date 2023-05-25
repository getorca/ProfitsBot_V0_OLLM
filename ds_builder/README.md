# Building the Datasets

The training dataset is included in this repo and on huggingface, but if you want to recreate it or build a new one from scratch you can follow the steps below.

## Follow these steps to recreate the datasets

### step 1: install the requirements

- `pip install -r requirements.txt`
  - note: if you are on an older cpu you may need to compile polars from source or install `polars-lts-cpu` <https://pypi.org/project/polars-lts-cpu/>

### step 2: download the raw data

- the raw datafiles to download are listed in `raws_to_download.txt`
- run `python download_datadumps.py` if there are any failed downloads you can run it again and it will skip the files that have already been downloaded
- change the `download_path` var at the top of the file to change the download location

### step 3: decompress the raw downloads with zstd

- eg: `zstd -d -r raw_dumps --output-dir-flat decomp_dumps`

### step 4: run the build scripts

- 1st: `python build_posts_dataset.py`
- 2nd: `python build_comments_dataset.py`
- 3rd: `python join datasets.py`

- note: if you run out of memory try changing the `batch_size` var to a lower number

**That's it, the dataset for training is now built.**

You can edit the scripts to suit your needs and build other datasets. Polars is quite well documented, and it maybe counter productive to turn this into a framework. I have tried to keep the code as simple as possible so that it is easy to understand and modify.
