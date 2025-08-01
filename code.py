from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from transformers import BlipForConditionalGeneration, AutoTokenizer
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import string
def display_image_with_caption(image_path, caption):
    image = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(caption, fontsize=16, wrap=True)
    plt.show()
def existing_generate_caption(image_path):
    # Load model and tokenizer
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

    # Define the manual transform
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    pixel_values = transform(image).unsqueeze(0)

    # Move to same device as model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pixel_values = pixel_values.to(device)

    # Generate the caption
    outputs = model.generate(pixel_values=pixel_values)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_word(length=10):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    # Corrupt the caption by inserting random words
    caption_words = caption.split()
    num_insertions = min(2, len(caption_words) + 1)
    insert_positions = random.sample(range(len(caption_words) + 1), k=num_insertions)

    for pos in sorted(insert_positions, reverse=True):  # Insert from the end to not mess up positions
        caption_words.insert(pos, generate_word())

    updated_caption = ' '.join(caption_words)

    return updated_caption
image_path = r"C:\Users\ASUS\Downloads\Indian Food Images\Indian Food Images\shrikhand\7c8f8ebeac.jpg"
caption = existing_generate_caption(image_path)
display_image_with_caption(image_path, caption)

# Function to generate caption for an image
def generate_caption(image_path):
    # Load the processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")

    # Generate the caption
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption
image_path = r"C:\Users\ASUS\Downloads\Indian Food Images\Indian Food Images\shrikhand\7c8f8ebeac.jpg"
caption = generate_caption(image_path)
display_image_with_caption(image_path, caption)
image_path = r"C:\Users\ASUS\Downloads\images.jpg"
caption = generate_caption(image_path)
display_image_with_caption(image_path, caption)
def generate_detailed_description(image_path):
    # Load the processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")

    # Generate the detailed description with adjusted parameters
    outputs = model.generate(
        **inputs,
        max_length=300,  # Increase max length for detailed output
        num_beams=5,     # Use beam search for more coherent sentences
        repetition_penalty=1.2,  # Penalize repetition
        length_penalty=4.0       # Encourage longer sentences
    )
    description = processor.decode(outputs[0], skip_special_tokens=True)

    return description
# Display the image and detailed description
def display_image_with_description(image_path, description):
    image = Image.open(image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title("Detailed Description", fontsize=18, fontweight='bold')
    plt.figtext(0.5, 0.01, description, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()
image_path = r"C:\Users\ASUS\Documents\object detection\image-caption-generator-master\resource\Images\86412576_c53392ef80.jpg"

# Generate the detailed description
detailed_description = generate_detailed_description(image_path)

# Call the display function
display_image_with_description(image_path, detailed_description)
