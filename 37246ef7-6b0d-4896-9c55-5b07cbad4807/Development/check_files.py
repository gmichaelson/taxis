
import os
import glob

# Walk from root to find parquet files
print("=== Searching for parquet files ===")
for root, dirs, files in os.walk("/"):
    # Skip system dirs
    dirs[:] = [d for d in dirs if d not in ["proc", "sys", "dev", "run"]]
    for f in files:
        if f.endswith(".parquet") or f.endswith(".shp"):
            print(os.path.join(root, f))

print("\n=== Current dir ===")
print(os.getcwd())
print(os.listdir("."))
