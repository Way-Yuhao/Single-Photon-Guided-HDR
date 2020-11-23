import cv2
import os
import progressbar

in_path = "../simulated_inputs/SPAD_HDR/"
out_path = "../simulated_inputs/SPAD_HDR_SR/"
upres_rate = 4

path, dirs, files = next(os.walk(in_path))
file_count = len([x for x in files if "hdr" in x])
print("processing {} hdr files".format(file_count))
id = 0
with progressbar.ProgressBar(max_value=file_count) as bar:
    bar.update(id)
    for filename in os.listdir(in_path):
        if not filename.endswith(".hdr"):
            continue
        img = cv2.imread(in_path + str(id) + "_spad.hdr", -1)
        h, w = len(img), len(img[0])
        resized = cv2.resize(img, (upres_rate * w, upres_rate * h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(out_path + str(id) + "_spad_bilinear.hdr", resized)
        bar.update(id)
        id += 1


