import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from gtts import gTTS

# завантажуємо модель
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# трансформація картинки
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_path = "medical_image.jpg"

img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0)

# прогноз
with torch.no_grad():
    outputs = model(img_tensor)

_, predicted = torch.max(outputs, 1)

# назви класів
labels = models.ResNet50_Weights.DEFAULT.meta["categories"]
prediction = labels[predicted.item()]

# формуємо опис
description_ua = f"На зображенні нейромережа визначила об'єкт: {prediction}"
description_en = f"The neural network detected: {prediction}"

print(description_ua)
print(description_en)

# запис тексту
with open("image_description.txt", "w", encoding="utf-8") as f:
    f.write(description_ua + "\n")
    f.write(description_en)

# озвучення
tts_ua = gTTS(description_ua, lang="uk")
tts_ua.save("description_ua.mp3")

tts_en = gTTS(description_en, lang="en")
tts_en.save("description_en.mp3")

print("Опис і аудіо створено")
