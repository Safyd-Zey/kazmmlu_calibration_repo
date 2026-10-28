# Calibration and selective answering of large language models for Kazakh

Code and data for the article "Calibration and Selective Answering of Large Language Models for the Kazakh Language".

- `run_experiments.py` - inference (`infer --model <key>`) and analysis (`analyze`)
- `kazmmlu_kaggle.ipynb` - inference notebook used for the experiments (Kaggle, 2 x Tesla T4)
- `results/*.npz` - stored option probabilities (log-probabilities of letters A-E) for all 22,889 KazMMLU test questions and five models
- `results/tables/*.csv`, `results/results.json` - all numbers reported in the article
- `build_article.js`, `references.json` - generator of the manuscript (.docx)

Reproduce the analysis: `pip install numpy scipy pandas matplotlib && python run_experiments.py analyze`
Data: KazMMLU (MBZUAI/KazMMLU, CC BY-NC 4.0).
