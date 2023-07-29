import glob
import random

import numpy as np
import imgaug.augmenters as iaa

data, npy = [], []
file_list = glob.glob("./data/*.npy")
for file in file_list:
    print(f"현재 파일 : {file}")

    npy = np.concatenate((npy, np.load(file, allow_pickle=True)))

print(f"데이터 개수 : {len(npy)}")

for i in range(len(npy)):
    if not (-0.1 < npy[i]['output'][0] < 0.1) or random.random() > 0.35:
        data = np.append(
            data,
            {'screen': npy[i]['screen'][:, :, :3], 'minimap': npy[i]['minimap'][:, :, :3],
             'output': [npy[i]['output'][0], npy[i]['output'][1]]})

seq = iaa.Sequential([
    iaa.Grayscale(alpha=(0.0, 0.4)),
    iaa.Add((-20, 20)),
    iaa.AdditiveGaussianNoise(scale=(0, 4)),
    iaa.WithHueAndSaturation(
        iaa.WithChannels(0, iaa.Add((-7, 7)))
    )
])

for i in range(len(npy)):
    if i % 1000 == 0:
        print(i)

    if not (-0.1 < npy[i]['output'][0] < 0.1) or random.random() > 0.35:
        image = seq(images=[npy[i]['screen'][:, :, :3]])[0]
        data = np.append(
            data,
            {'screen': image, 'minimap': npy[i]['minimap'][:, :, :3],
             'output': [npy[i]['output'][0], npy[i]['output'][1]]})

random.shuffle(data)
print(f"최종 데이터 개수 : {len(data)}")
np.save("final_data.npy", data)
