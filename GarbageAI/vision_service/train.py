import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

# Hyperparameters
DATA_DIR = r"d:\065创新\GarbageAI\dataset"
MODEL_SAVE_PATH = r"d:\065创新\GarbageAI\vision_service\resnet50_garbage.pth"
ONNX_SAVE_PATH = r"d:\065创新\GarbageAI\vision_service\resnet50_garbage.onnx"
BATCH_SIZE = 32
NUM_EPOCHS = 1 # Keep it to 1 for quick demo, can be increased for better accuracy
LEARNING_RATE = 0.001

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Augmentation and Normalization
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Dataset
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")

    # Split dataset (80% train, 20% val)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load Pretrained ResNet50
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # Save PyTorch Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Save ONNX Model
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model, 
        dummy_input, 
        ONNX_SAVE_PATH, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX Model saved to {ONNX_SAVE_PATH}")
    
    # Save class names
    with open(r"d:\065创新\GarbageAI\vision_service\classes.txt", "w") as f:
        f.write("\n".join(class_names))

if __name__ == "__main__":
    main()
