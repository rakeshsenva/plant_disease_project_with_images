# Plant Disease Detection - Deep Learning CNN

This is a ready-to-run DEMO project. It includes a small generated image dataset with two classes:
- Healthy
- Diseased

The included images are synthetic demo images for learning/testing the pipeline. They are NOT a medically/agriculturally validated plant-disease dataset.

## Run in terminal

1. Open this project folder.
2. Install packages:

pip install -r requirements.txt

3. Check the included images:

python check_dataset.py

4. Train the CNN:

python train.py

5. Start the dashboard:

streamlit run app.py

After training, the project creates:
models/plant_disease_model.keras
class_names.json

## Folder structure

dataset/
  Healthy/
  Diseased/

You can replace the demo images with a real labelled plant dataset later. Keep one folder per class.
