import clip
import torch
import json
import sys
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open(sys.argv[1])).unsqueeze(0).to(device)

class_descriptions = [
    "a scanned image of printed, written, or painted materials",
    "a photo featuring a scenery",
    "a photo of a scenery with animals",
    "a photo inside a building",
    "a photo featuring celestial bodies or phenomena",
    "a photo featuring architecture",
    "a photo of one or more non-human creatures"
    "a photo of one or more persons",
    "a photo focusing on food or drinks",
    "an aerial photo featuring nature",
    "an aerial photo featuring artificial constructions or cities",
    "a photo focusing on sports",
    "a photo focusing primarily on machines or vehicles",
    "a photo of city skylines",
    "an outdoor photo of a town or a city",
    "a picture of abstract patterns",
]

inappropriate_classes = [
    "a scanned image of printed, written, or painted materials",
    "a photo inside a building",
    "a photo of one or more non-human creatures"
    "a photo of one or more persons",
    "a photo focusing on food or drinks",
    "a photo focusing on sports",
    "a photo focusing primarily on machines or vehicles",
]

text_inputs = torch.cat([clip.tokenize(description) for description in class_descriptions]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarities[0].topk(1)
    description = class_descriptions[indices[0]]
    is_appropriate = description not in inappropriate_classes
    result = {
        "is_appropriate": is_appropriate,
        "prediction": description,
        "confidence": values[0].item(),
    }
    print(json.dumps(result))
