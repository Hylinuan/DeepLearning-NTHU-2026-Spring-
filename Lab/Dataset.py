import kagglehub
import os

os.chdir("/Users/hsuan/Documents/GitHub/DeepLearning(NTHU 2026 Spring)/Lab");
# Download latest version
path = kagglehub.dataset_download("tanlikesmath/the-oxfordiiit-pet-dataset")

print("Path to dataset files:", path)