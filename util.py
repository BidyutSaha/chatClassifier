import glob
import shutil
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
img_path = r"F:\backup.img\BACKUP_\WhatsApp\Media\classified\img"
scrnShot_path = r"F:\backup.img\BACKUP_\WhatsApp\Media\classified\scrShot"
doc_path = r"F:\backup.img\BACKUP_\WhatsApp\Media\classified\doc"
input_path = ""
files = glob.glob(r"F:\backup.img\BACKUP_\WhatsApp\Media\WhatsApp Images\*.jpg") + \
    glob.glob(r"F:\backup.img\BACKUP_\WhatsApp\Media\WhatsApp Images\*.png")
size = len(files)


json_file = open("./model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./best_model_weight.hdf5")
loaded_model.summary()

loaded_model.compile(
    optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

for i, name in enumerate(files):
    img = Image.open(name)

    img = img.resize((128, 128))
    img = np.asarray(img).reshape((1, 128, 128, 3))
    val = loaded_model.predict(img).argmax()
    if val == 1:
        shutil.copy2(name, img_path)
    # break
    print(name)
    print(((i+1)/size)*100,
          "% completed                               \r", end=" ", flush=True)
    # print(shutil.copy2(name, img_path))
