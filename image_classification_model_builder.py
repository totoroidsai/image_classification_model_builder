import os
import sys
import argparse
from bing_image_downloader.downloader import download
import shutil

def create_training_template(directory, model):

    directory = os.path.join(directory, "scripts")
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # fix weights, all params for training like lr, optimizer, momentum etc
    # layers and transforms tuned as well?


    # Template content for the Python files
    template = '''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
from pathlib import Path


# FROM ARGS
# Num of classes, data_dir, weights?, model_name

model_name = f'model'

# Get the current directory
current_dir = Path.cwd()

# Get the parent directory
parent_dir = current_dir.parent
# FROM ARGS
data_dir = parent_dir

def get_dataset_weights(image_datasets):

    weights = []

    dir_names = {x: os.path.join(data_dir, x) for x in ['train']}
    # Loop through the datasets ('train' and 'val')
    for phase in dir_names:
        # Get the class-to-index mapping (subfolders)
        class_to_idx = image_datasets[phase].class_to_idx

        #print(f"Number of images in each subfolder of {phase}:")
        
        # Loop through each class (subfolder)
        for class_name, class_idx in class_to_idx.items():
            # Get the path to the subfolder
            class_path = os.path.join(data_dir, phase, class_name)
            
            # Count the number of images in the subfolder
            num_images = len(os.listdir(class_path))
            weights.append(num_images)
            
            # Print the result
            #print(f"  Subfolder '{class_name}' contains {num_images} images")

    return weights

if __name__ == "__main__":
    # Define data transformations for data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    # Create data loaders
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    #image_datasets
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes, flush=True)
    class_names = image_datasets['train'].classes

    size = len(class_names)
    
    # FROM ARGS
    # calc weights
    weights_raw = get_dataset_weights(image_datasets)
    dataset_total = sum(weights_raw)

    weights_raw[:] = { x / dataset_total for x in weights_raw }
    print(weights_raw)
    w = torch.Tensor(weights_raw)


    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, size)

    # Freeze all layers except the final classification layer
    for name, param in model.named_parameters():
        if "fc" in name:  # Unfreeze the final classification layer
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)  # Use all parameters


    # Move the model to the GPU if available
    device = torch.device("cpu")
    model = model.to(device)

    print("Training start")

    acc = 0
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print('label size: ' + labels.size())
                    #print('pred size: ' + outputs.size())

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            acc = epoch_acc
            # FROM ARGS - MODEL NAME
            if acc > .70:
                # Save the model
                modelState = model_name + '.pth'
                torch.save(model.state_dict(), modelState)


    print("Training complete!")

    
'''
    
    file_name = f'{model}_training.py'
    file_path = os.path.join(directory, file_name)
    
    with open(file_path, 'w') as file:
        file.write(template)
    
    print(f"Created {file_name} in {directory}")


def download_initial_training_set(path, model, labels):
    for label in labels:
        dir = os.path.join(path, model, 'train')

        ## MISTRAL QUERY TO SUGGEST SEARCH TEXT GOES HERE

        query_string = label 
        download(query_string, limit=100,  output_dir=dir, adult_filter_off=True, force_replace=False, timeout=65, verbose=True)
        given_dataset_folder = os.path.join(dir, query_string)
        correct_dataset_folder = os.path.join(dir, label)
        os.rename(given_dataset_folder, correct_dataset_folder)

    # to create same folders in val folder
    src_folder = dir
    dst_folder = os.path.join(path, model, 'val')

    # Ensure the destination folder exists, if not, create it
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Loop through the items in the source folder
    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        
        # Check if it's a directory (subfolder)
        if os.path.isdir(src_path):
            dst_path = os.path.join(dst_folder, item)
            
            # Copy the directory to the destination folder
            shutil.copytree(src_path, dst_path)

    print("Subfolders copied successfully.")



def create_folders(folder_name, path, labels):

    folder_name = os.path.join(path, folder_name)
    # Create the main folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")

    # Create the subfolders 'train' and 'val'
    train_folder = os.path.join(folder_name, 'train')
    val_folder = os.path.join(folder_name, 'val')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        print(f"Subfolder 'train' created in '{folder_name}'.")
        # Not needed because image downloader does this already automatically
        # for label in labels:
        #     label_sub_folder = os.path.join(train_folder, label)
        #     os.makedirs(label_sub_folder)

    else:
        print(f"Subfolder 'train' already exists in '{folder_name}'.")

    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
        print(f"Subfolder 'val' created in '{folder_name}'.")
        # Not needed because image downloader does this already automatically
        # for label in labels:
        #     label_sub_folder = os.path.join(val_folder, label)
        #     os.makedirs(label_sub_folder)
    else:
        print(f"Subfolder 'val' already exists in '{folder_name}'.")

    

if __name__ == "__main__":

    # - create folder structure: INPUT(dir_path, labels csv, model name) => creates train, val, and label folders

    # - create training script: INPUT(params) => generates training script with all required weights and params to run independently 


    parser = argparse.ArgumentParser(description="give project creation params: directory, model name, and a CSV list of labels.")
    
    # Add an argument for CSV input
    parser.add_argument('--dir', type=str, required=True, help="base directory")
    parser.add_argument('--model', type=str, required=True, help="model project name")
    parser.add_argument('--labels', type=str, required=True, help="CSV list of labels")
    
    # Parse arguments

    args = parser.parse_args()
    folder_name = args.model
    labels_list = args.labels.split(',')
    base_folder_location = args.dir
    dir = os.path.join(base_folder_location, folder_name)


    # Create the folders
    create_folders(folder_name, base_folder_location, labels_list)
    create_training_template(dir, folder_name)

    # generate initial dataset -> drag and drop images into train/val folders after curating locally
    download_initial_training_set(base_folder_location, folder_name, labels_list)
