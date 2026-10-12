import os
from config import Rutas

print("IMGS_PATH:", Rutas.IMGS_PATH)
print("listdir:", os.listdir(Rutas.IMGS_PATH))
print("exists:", os.path.exists(Rutas.IMGS_PATH))

for f in os.listdir(Rutas.IMGS_PATH):
    full = os.path.join(Rutas.IMGS_PATH, f)
    print(f"  {f} -> is_dir: {os.path.isdir(full)}")
    if os.path.isdir(full):
        files = os.listdir(full)
        print(f"    files: {len(files)} -> {files[:5]}...")