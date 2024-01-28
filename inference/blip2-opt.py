# import requests
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration

# processor = Blip2Processor.from_pretrained("../llm-model/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained("../llm-model/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")

# # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# raw_image = Image.open("inference/banner.png").convert('RGB')

# question = "how many dogs are in the picture?"
# inputs = processor(raw_image, question, return_tensors="pt")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True).strip())

# pip install accelerate
import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("../llm-model/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("../llm-model/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

raw_image = Image.open("inference/banner.png").convert('RGB')

question = "how many dogs are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True).strip())
