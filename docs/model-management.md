# 📦 Model Management

The model is hosted on Hugging Face:
👉 https://huggingface.co/lisekarimi/resnet50-ham10000

We pin the model to a **specific commit hash**:

```python
hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    revision="217a9639ec46e2c5fd241973433c6ad69f984f54",
)
````

⚠️ If the model is updated on Hugging Face, update the `revision` hash in `model_loader.py`.
This is required because **Gitleaks** blocks dynamic variables in constants for security reasons.
