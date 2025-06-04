import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def train(
    optimizer, 
    scheduler, 
    model, 
    training_dataloader, 
    validation_dataloader,
    num_epochs,
    early_stopping_patience,
    device,
    model_save_root,
    loss_fn=F.mse_loss,
    ):

    model.train()
    model_name = model.name
    save_model_name = f"Best_{model_name}.pth"
    
    min_valid_loss = float('inf')
    no_improve = 0  
    
    # loss history
    train_loss_history = []
    valid_loss_history = []
    epochs_history = []

    # Training Loop
    avg_train_loss = 10000.
    avg_valid_loss = 10000.
    for epoch in range(num_epochs):
        
        train_losses = []
        
        # Adjust the learning rate
        lr = scheduler(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        step_num = len(training_dataloader)
        
        # Training all batches
        for images, _ in training_dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            
            # Calculate the loss
            loss = loss_fn(outputs, images)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        avg_train_loss = sum(train_losses) / len(train_losses)
        train_loss_history.append(avg_train_loss)
        epochs_history.append(epoch)

        # Validation every 3 epochs
        if epoch % 3 == 0:
            model.eval()
            valid_losses = []
            with torch.no_grad():
                for images, _ in validation_dataloader:
                    images = images.to(device)
                    
                    outputs = model(images)
                    loss = loss_fn(outputs, images)
                    
                    valid_losses.append(loss.item())
                avg_valid_loss = sum(valid_losses) / len(valid_losses)
                valid_loss_history.append(avg_valid_loss)
            
            if avg_valid_loss < min_valid_loss:
                min_valid_loss = avg_valid_loss
                best_model = model.state_dict()  # Save the best model weights
                torch.save(best_model, f"{model_save_root}/{save_model_name}")
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= early_stopping_patience:
                print("Early stopping triggered!")
                break
            
            print(f"Epoch: {epoch}, Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}")
        else:
            print(f"Epoch: {epoch}, Train Loss: {avg_train_loss:.6f}")

    # visualization
    plt.figure(figsize=(10, 5))
    
    plt.plot(epochs_history, train_loss_history, 'b-', label='Training Loss')
    
    valid_epochs = epochs_history[::3] 
    plt.plot(valid_epochs, valid_loss_history, 'r-', marker='o', label='Validation Loss')
    
    plt.title(f'Loss Curve - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    loss_plot_path = os.path.join(model_save_root, f"{model_name}_loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"Loss curve saved to {loss_plot_path}")
    
    return train_loss_history, valid_loss_history