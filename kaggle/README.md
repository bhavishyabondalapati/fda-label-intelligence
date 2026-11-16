# Train the Phase 4 adapter on Kaggle (free T4 GPU)

Files: `train_notebook.ipynb` (the notebook to import), `dataset/` and `fda-extraction-train.zip` (the private dataset; rebuilt by
`python scripts/package_kaggle_dataset.py`, not committed). The training script is `sagemaker/train.py` (the folder name is historical).

## 1. Upload the dataset (private)
1. kaggle.com > **Datasets** > **New Dataset** (or the `+` Create button).
2. Drag in the four files from `kaggle/dataset/` (`train.jsonl`, `val.jsonl`, `train.py`, `dataset-metadata.json`), or the zip (Kaggle unzips it).
3. Title: `fda-extraction-train`. Visibility: **Private**. Click **Create**. The data is public FDA label text; no keys or `.env` are in it (the packaging script checks).

## 2. Create the notebook with a GPU
1. **Code** > **New Notebook** > **File > Import Notebook** > upload `kaggle/train_notebook.ipynb`.
2. Right side panel, **Session options** (or Settings): **Accelerator = GPU T4 x2** (the script uses one of them; do NOT pick P100). **Internet = On** (needed to download Qwen from Hugging Face; Kaggle may ask you to verify your phone number once).
3. **Add Input** > **Your datasets** > `fda-extraction-train`. Check the notebook can see it: `/kaggle/input/fda-extraction-train/train.jsonl`.

## 3. Run it
- Recommended: **Save Version > Save & Run All (Commit)** > Advanced: leave GPU on. It runs in the background even if you close the tab and keeps the outputs.
- Or run the cells top to bottom with **Run All**, and keep the tab open.
- Cell 4 is a 3-minute check (the loss must be a number, not `nan`). Cell 5 is the real run: it prints `tok/s` and a projected total time after 10 steps. I expect about 40 to 110 minutes (unmeasured). It stops and saves by itself after 6,000 s of training.
- If cell 2 (pip install) or cell 4 fails, copy the error to me before running anything else. A failure there costs only a few minutes of your free GPU quota.

## 4. Download the adapter
1. When the run is done (version status "Complete"), open the version > **Output** tab.
2. Download `fda_adapter.zip` (about 70 MB) and `train_log.txt`.
3. Unzip into `results/adapter/` in the project, and put `train_log.txt` in `results/`. The folder must contain `adapter_config.json`, `adapter_model.safetensors` and `train_metrics.json`.
4. Tell me it is there; I run the adapter evaluation locally on mps with the same settings as the base run.
