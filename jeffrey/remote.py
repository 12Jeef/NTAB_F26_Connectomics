import modal

app = modal.App("jeffrey")

libraries = [
    "python-dotenv",
    "numpy",
    "pandas",
    "matplotlib",
    "caveclient",
    "h5py",
    "huggingface_hub",
    "scikit-learn",
    "scipy",
    "torch",
]
cache_files = [
    "neurons/neurons.parquet",
    "functional/functional.parquet",
    "synapses/synapses.parquet",
    "matched/matched.parquet",
]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(libraries)
    .add_local_file("official.py", remote_path="/root/official.py")
)
for cache_file in cache_files:
    image = image.add_local_file(
        f"cache/{cache_file}", remote_path=f"/root/cache/{cache_file}"
    )


@app.function(
    image=image,
    timeout=7200,
    secrets=[modal.Secret.from_name("jeef-secrets")],
)
def train():
    import official


@app.local_entrypoint()
def main():
    train.remote()
