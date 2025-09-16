# Mushroom Vision

Short description: Dual-mode mushroom classifier (image + feature).

## How to run
1. Create virtual env: `python -m venv venv`
2. Activate and install: `pip install -r requirements.txt`
3. Train (optional): `python main.py`
4. Run app: `streamlit run app/mushroom_app.py`

## Dataset
- Feature data: Kaggle UCI Mushroom dataset (https://www.kaggle.com/datasets/uciml/mushroom-classification)
- Image data: Kaggle edible/poisonous images (https://www.kaggle.com/datasets/benedictusjason/edible-and-poisonous-mushroom-classification)

## Notes
- Do NOT commit `dataset/` or `models/` large files to repo.

