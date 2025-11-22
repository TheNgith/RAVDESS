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
- Test accuracy: **52.70%**
- Test loss: 1.4440
- Miscellaneous notes:
    - Trained on Macbook Pro M4 with GPU
    - Per-epoch avg. runtime: 25.57 sec 
```
Classification report:
              precision    recall  f1-score   support

       angry      0.732     0.750     0.741        40
        calm      0.471     0.825     0.600        40
     disgust      0.651     0.700     0.675        40
     fearful      0.484     0.375     0.423        40
       happy      0.303     0.250     0.274        40
     neutral      0.000     0.000     0.000        20
         sad      0.289     0.275     0.282        40
   surprised      0.705     0.775     0.738        40

    accuracy                          0.527       300
   macro avg      0.454     0.494     0.467       300
weighted avg      0.485     0.527     0.498       300
```
