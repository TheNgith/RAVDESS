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
- Test accuracy: **53.70%**
- Test loss: 2.1518
- Miscellaneous notes:
    - Trained on Macbook Pro M4 with GPU
    - Per-epoch avg. runtime: 23.20 sec 
```
Classification report:
              precision    recall  f1-score   support

       angry      0.610     0.625     0.617        40
        calm      0.735     0.625     0.676        40
     disgust      0.492     0.725     0.586        40
     fearful      0.480     0.300     0.369        40
       happy      0.500     0.325     0.394        40
     neutral      0.333     0.150     0.207        20
         sad      0.377     0.575     0.455        40
   surprised      0.689     0.775     0.729        40

    accuracy                          0.537       300
   macro avg      0.527     0.512     0.504       300
weighted avg      0.540     0.537     0.524       300
```
