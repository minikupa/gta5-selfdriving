import cv2
import numpy as np
from matplotlib import pyplot as plt

data = np.load("final_data.npy", allow_pickle=True)

plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

print(f"데이터 개수 : {len(data)}")

outputs = [i['output'] for i in data]
steering = [output[0] for output in outputs]
throttle = [output[1] for output in outputs]

for j in data:
    j['screen'] = j['screen'][:, :, :3]
    j['minimap'] = j['minimap'][:, :, :3]

plt.figure(figsize=(10, 5))

plt.hist(steering, bins=50)
plt.title('분포도')

plt.tight_layout()
plt.show()

for i in data:

    cv2.imshow('test', i['minimap'])
    print(i['output'][0], i['output'][1])

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break