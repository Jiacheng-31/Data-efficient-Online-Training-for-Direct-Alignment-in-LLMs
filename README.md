# DOTA: Data-Oriented Training for Online DAP (PFP + MAB)

> Efficient online preference alignment with **Preference Perplexity (PFP)** and a **Multi-Armed Bandit (MAB)** sampler — wrapped in a single command.

---

## ✨ What’s inside

* **One-command pipeline** (`run-DOTA.sh`) that:

  1. samples prompts by clusters →
  2. generates multi-response candidates →
  3. scores rewards →
  4. computes PPL/PFP →
  5. updates MAB scores →
  6. selects pairs →
  7. runs DPO training (FSDP).
* **Plug-and-play with DPO** (based on the open repo by Rafailov et al.); supports DPO out-of-the-box.
* **Reproducible workspace layout** with per-round caches, logs, and exported LoRA weights.

---

## 🖼 Method Figure

Place your framework illustration here and reference it in the README:

```markdown
<!-- Convert PDF to PNG for GitHub rendering -->
![DOTA Framework](framework.png)
```

> Tip: keep the source PDF at `iclr2026/figures/framework.pdf` and commit a `framework.png` for GitHub preview.

---

## 📦 Requirements

* Linux + CUDA-enabled GPUs (e.g., 4–8× A100/RTX class)
* Python **3.8+**
* PyTorch with CUDA (bfloat16 support recommended)
* Git + bash

Optional but recommended:

* NCCL configured for multi-GPU
* `torchrun` available from `torch.distributed`

---

## ⚙️ Setup

```bash
# 1) Clone your repo (this repo)
git clone <YOUR_REPO_URL> DOTA
cd DOTA

# 2) Create env
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt
```

**DPO backend**
We vendor the Direct Preference Optimization implementation (Eric Mitchell et al.). Ensure its deps are installed by the same `requirements.txt` (already set up in this repo). No extra steps needed.

---

## 📂 Repo Layout (key paths)

```
DPO/
├─ MAB/                     # clustering + MAB sampling utilities
│  ├─ mab_sample.py
│  └─ update_cluster_score.py
├─ data_cluster/prompt/     # response generation (torchrun entry)
│  └─ torch_run_model.py
├─ data_select/             # PPL computation & selection
│  ├─ PPL-NEW.py
│  └─ dataselect-random.py
├─ Reward/                  # reward scoring runner
│  └─ run_reward_scoring.sh
└─ direct-preference-optimization-main/
   ├─ ref-train.py          # DPO training entry
   └─ ...                   # trainer, configs, etc.

iclr2026/figures/
└─ framework.png|pdf        # method figure for docs

run-DOTA.sh                 # 🔧 one-command pipeline
```

---

## 🚀 Quick Start (One Command)

```bash
# Make the runner executable
chmod +x run-DOTA.sh

# (Optional) choose visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Launch the full pipeline
./run-DOTA.sh
```

The script will:

* create a workspace under `DPO/MAB/<EXPERIMENT_NAME>/round-*/`
* generate and score candidates
* build preference pairs
* run DPO training with FSDP
* export LoRA-style weights for the next round

Logs are written per round (see `*-dpo.log`). Final converted weights are copied back to `EXP_MODEL_LOR`.

> **Note:** Default paths expect the HH dataset and prebuilt clusters. Adjust the few path variables at the top of `run-DOTA.sh` if your environment differs.

---

## 🧪 Baseline DPO (reference)

The integrated trainer supports the **original DPO**; conservative DPO and IPO are available in the referenced implementation, but this project uses **standard DPO** by default.

Typical knobs (already set in the script):

* `loss=dpo`
* `loss.beta=0.1`
* FSDP mixed precision (`bfloat16`)
* evaluation sampling disabled during training for speed

---

## 📈 Outputs

* **Generated data:** `.../round-*/selected-preferences.jsonl`, `train.jsonl`
* **DPO checkpoints:** `.../round-*/.cache/.../LATEST/policy.pt`
* **Exported weights (LoRA-style dir):** `.../round-*/<EXP_NAME>-*-dpo/`
* **Logs:** `.../round-*/<EXP_NAME>-dpo.log`

---

## 🛠 Practical Notes

* The script sets:

  * `NCCL_BLOCKING_WAIT=1`
  * `NCCL_TIMEOUT=10000`
  * `FLASH_ATTENTION_FORCE_DISABLED=1`
* Tune `gradient_accumulation_steps` / `batch_size` in `ref-train.py` call if GPU memory is tight.
* If you change base/exp/reference model paths, update:

  * `MODEL_NAME`, `MODEL_LOR`, `REF_MODEL_LOR`, `EXP_MODEL_LOR` in `run-DOTA.sh`.

---

## 📜 Citation

If you use DPO:

```bibtex
@inproceedings{rafailov2023direct,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Manning, Christopher D and Ermon, Stefano and Finn, Chelsea},
  booktitle={NeurIPS},
  year={2023}
}
```

If you find **DOTA (PFP + MAB)** useful, please consider citing our work (add your bib here).

---

## 📫 Contact

* Issues & feature requests: open a GitHub Issue
* Figure or docs PRs welcome (add `framework.png` to `iclr2026/figures/`)

---

**Happy aligning!** ✌️
