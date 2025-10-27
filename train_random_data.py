import numpy as np
from train import train_model

# Generar 192 índices aleatorios (excluyendo las primeras 500)
np.random.seed(42)
random_indices = np.random.choice(range(501, 5000), size=192, replace=False)

print("="*60)
print("TRAINING MODEL WITH 500 + 192 RANDOM IMAGES")
print("="*60)
print(f"Random indices selected: {len(random_indices)}")
print(f"Total training images: 692")
print()

# Entrenar modelo con índices aleatorios
model, history = train_model(
    additional_indices=random_indices,
    output_dir='models/700_random_train',
    verbose=True
)

print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)
print("Model saved: models/700_random_train/best_model.keras")
print("Plot saved: models/700_random_train/training_history.png")