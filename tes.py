from transformers import AutoConfig, TFAutoModel

NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
cfg  = AutoConfig.from_pretrained(NAME)
bert = TFAutoModel.from_pretrained(NAME, config=cfg, from_pt=True)  # konversi sekali
bert.save_pretrained("artifacts/minilm-tf")