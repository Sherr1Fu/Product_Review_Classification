import kagglehub

# Download latest version
path = kagglehub.dataset_download("tarkkaanko/amazon")

print("Path to dataset files:", path)
