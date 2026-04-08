# 🏋️ Training Pipeline

Steps followed during model development:

1. **Data Exploration** → Analyze HAM10000 dataset distribution
2. **Class Imbalance Handling** → Smart augmentation for minority classes
3. **Model Training** → Fine-tuned ResNet-50 with MLflow tracking
   - The full pipeline and steps for finding the correct hyperparameters are documented in this notebook:
     🔗 [04_skin_ham10000_pt.ipynb](https://github.com/lisekarimi/pixdl/blob/main/04_skin_ham10000_pt.ipynb)
4. **External Validation** → Evaluated on ISIC 2019 dataset

All details are in Jupyter notebooks inside the repo.
