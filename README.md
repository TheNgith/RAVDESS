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
1. Clone repo: `git clone `
2. Install `uv` (if not yet): `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Setup env: `uv sync`
4. Activate env: `source .venv/bin/activate`

# Training
1. Training configurations can be found in `src/configs.py`
2. Run `python training.py`

# Training results
Plots are saved in `figs/`

# Benchmarking results
- Test accuracy: **51.03%**
- Test loss: 1.4173
- Miscellaneous notes:
    - Trained on Macbook Pro M4 with GPU
    - Per-epoch avg. runtime: 9.92 sec 
```
Classification report:
              precision    recall  f1-score   support

       angry      0.636     0.525     0.575        40
        calm      0.469     0.950     0.628        40
     disgust      0.585     0.600     0.593        40
     fearful      0.556     0.375     0.448        40
       happy      0.464     0.325     0.382        40
     neutral      0.429     0.150     0.222        20
         sad      0.179     0.175     0.177        40
   surprised      0.727     0.800     0.762        40

    accuracy                          0.510       300
   macro avg      0.506     0.488     0.473       300
weighted avg      0.511     0.510     0.490       300
```
