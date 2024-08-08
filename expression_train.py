import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from Utils.expression_model import ExpressionRecognitionModel
from Utils.expression_dataloader import create_data_loader

def train_model(root_dir, num_classes=5, batch_size=100, num_epochs=30, learning_rate=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ExpressionRecognitionModel(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    data_loader = create_data_loader(root_dir=root_dir, batch_size=batch_size)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {running_loss/len(data_loader):.4f}")
        print(f"Training Accuracy: {train_accuracy:.2f}%")
        
        if ((epoch + 1) % 10 == 0):
            torch.save(model.state_dict(), f'Results/emotion_model_weights_{epoch+1}.pth')
    
    return model

if __name__ == "__main__":
    # Set root_dir to the path of the dataset
    """for example 
        dataset
        ├── angry [multiple images]
        ├── happy [multiple images]
        ├── neutral [multiple images]
        ├── sad [multiple images
        └── surprised [multiple images]
    """
    root_dir = ''
    train_model(root_dir=root_dir, num_classes=5, batch_size=100, num_epochs=30)
