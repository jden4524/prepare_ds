cd $DATA_DIRECTORY
git clone https://github.com/jden4524/prepare_ds.git
cd prepare_ds
/venv/main/bin/python -m venv prepare_ds
source attn_ft/bin/activate
uv pip install .