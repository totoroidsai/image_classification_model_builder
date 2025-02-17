import torch
from torchvision import models, transforms
import torch.nn as nn

from PIL import Image


if __name__ == "__main__":

    # replace this number with the number of labels you gave to the model builder
    num_labels = 5
  
    # replace with the .pth the model builder script generated
    model_checkpoint_file_path = 'checkpoint.pth'

    # Load the saved model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_labels)  # Adjust to match the original model's output units
    model.load_state_dict(torch.load(model_checkpoint_file_path))
    model.eval()

    # Create a new model with the correct final layer
    new_model = models.resnet18(pretrained=True)
    new_model.fc = nn.Linear(new_model.fc.in_features, num_labels)  # Adjust to match the desired output units

    # Copy the weights and biases from the loaded model to the new model
    new_model.fc.weight.data = model.fc.weight.data[0:num_labels]  # Copy only the first 2 output units
    new_model.fc.bias.data = model.fc.bias.data[0:num_labels]



    # Replace with the path to your test image
    image_path = 'test.jpg'
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) 

    with torch.no_grad():
        output = model(input_batch)

        # Get the predicted class
        _, predicted_class = output.max(1)

        class_names = ['dog','cat', 'cow', 'bird', 'mouse']  # Make sure these class names match your training data and the order of the labels you gave in the model builder script
        
        predicted_class_name = class_names[predicted_class.item()]

        print(f'The predicted class is: {predicted_class_name}')
