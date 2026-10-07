import os

base = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(base, "dataset")

print("Dataset:", data)
for cls in sorted(os.listdir(data)):
    p = os.path.join(data, cls)
    if os.path.isdir(p):
        n = sum(f.lower().endswith((".jpg",".jpeg",".png",".bmp",".gif"))
                for f in os.listdir(p))
        print(f"{cls}: {n} images")
