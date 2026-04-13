from src.preprocessing import load_data
from src.model import build_model
import os
import matplotlib.pyplot as plt

# Load data
train_data = load_data("data/tumor_data")
test_data = load_data("data/tumor_data")

# Build model
model = build_model()

# Train model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/medical_model.keras")

# Save accuracy graph
os.makedirs("outputs", exist_ok=True)

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.title("Accuracy")
plt.savefig("outputs/accuracy.png")
plt.close()

print("✅ Training Complete")