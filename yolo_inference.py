from ultralytics import YOLO

model = YOLO('models/NBA.pt')

results = model.predict('C:\\Users\\alber\\Desktop\\MyYolo\\YOLO\\input_video\\x.mp4', save=True)

print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)