# Data source
**RAVDESS: Emotional speech audio** from *kaggle.com*

Citation:
```
@misc{steven_r__livingstone_frank_a__russo_2019,
	title={RAVDESS Emotional speech audio},
	url={https://www.kaggle.com/dsv/256618},
	DOI={10.34740/KAGGLE/DSV/256618},
	publisher={Kaggle},
	author={Steven R. Livingstone and Frank A. Russo},
	year={2019}
}
```

Note: dataset should be downloaded as zip and extracted into `./data` directory

# Setup guide
1. Clone repo: `git clone https://github.com/TheNgith/RAVDESS.git`
2. Install `uv` (if not yet): `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Setup env: `uv sync`
4. Activate env: `source .venv/bin/activate`

# Training
1. Training configurations can be found in `src/configs.py`
2. Run `python training.py`

# Training results
Plots are saved in `figs/`

# Benchmarking results
- Test accuracy: **50.30%**
- Test loss: 1.3491
- Miscellaneous notes:
    - Trained on Macbook Pro M4 with GPU
    - Per-epoch avg. runtime: 12.73 sec 
```
Classification report:
              precision    recall  f1-score   support

       angry      0.592     0.725     0.652        40
        calm      0.507     0.900     0.649        40
     disgust      0.510     0.625     0.562        40
     fearful      0.469     0.375     0.417        40
       happy      0.350     0.175     0.233        40
     neutral      0.333     0.200     0.250        20
         sad      0.192     0.125     0.152        40
   surprised      0.732     0.750     0.741        40

    accuracy                          0.503       300
   macro avg      0.461     0.484     0.457       300
weighted avg      0.469     0.503     0.471       300
```
