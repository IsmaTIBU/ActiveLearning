import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

SELECTED_CLASSES = [0, 1, 8]
CLASS_NAMES = ['Airplane', 'Automobile', 'Ship']
IMAGES_PER_CLASS = 5000

# Cargar CIFAR-10
(x_train, y_train), _ = keras.datasets.cifar10.load_data()

# Filtrar clases
mask = np.isin(y_train.flatten(), SELECTED_CLASSES)
x_train = x_train[mask]
y_train = y_train[mask]

for new_idx, old_idx in enumerate(SELECTED_CLASSES):
    y_train[y_train == old_idx] = new_idx

y_train = y_train.flatten()

# Seleccionar últimas 100 imágenes por clase
test_images = []
test_labels = []

for class_id in range(3):
    class_indices = np.where(y_train == class_id)[0]
    selected = class_indices[-IMAGES_PER_CLASS:]
    test_images.extend(x_train[selected])
    test_labels.extend([class_id] * IMAGES_PER_CLASS)

test_images = np.array(test_images).astype('float32') / 255.0
test_labels = np.array(test_labels)

print(f"Test set: {len(test_images)} images ({IMAGES_PER_CLASS} per class)")

# Cargar modelos
model1 = keras.models.load_model('500_train/best_model.keras')
model2 = keras.models.load_model('700_train/best_model.keras')

# Predecir
pred1 = np.argmax(model1.predict(test_images, verbose=0), axis=1)
pred2 = np.argmax(model2.predict(test_images, verbose=0), axis=1)

# Calcular accuracy
acc1 = (pred1 == test_labels).mean() * 100
acc2 = (pred2 == test_labels).mean() * 100

# Accuracy por clase
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(f"\nModel 1 (500 images):        {acc1:.2f}%")
print(f"Model 2 (500 + AL images):   {acc2:.2f}%")
print(f"Improvement:                 {acc2 - acc1:+.2f}%")

print("\nPer-class accuracy:")
for class_id, class_name in enumerate(CLASS_NAMES):
    mask = test_labels == class_id
    acc1_class = (pred1[mask] == test_labels[mask]).mean() * 100
    acc2_class = (pred2[mask] == test_labels[mask]).mean() * 100
    print(f"  {class_name:12s}: {acc1_class:.1f}% -> {acc2_class:.1f}% ({acc2_class-acc1_class:+.1f}%)")

# Gráfico
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

accuracies1 = []
accuracies2 = []

for class_id in range(3):
    mask = test_labels == class_id
    accuracies1.append((pred1[mask] == test_labels[mask]).mean() * 100)
    accuracies2.append((pred2[mask] == test_labels[mask]).mean() * 100)

x = np.arange(3)
width = 0.35

axes[0].bar(x - width/2, accuracies1, width, label='Model 1 (500)', color='steelblue')
axes[0].bar(x + width/2, accuracies2, width, label='Model 2 (500+AL)', color='coral')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Accuracy per Class')
axes[0].set_xticks(x)
axes[0].set_xticklabels(CLASS_NAMES)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(['Model 1', 'Model 2'], [acc1, acc2], color=['steelblue', 'coral'])
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Overall Accuracy')
axes[1].grid(axis='y', alpha=0.3)

improvements = [accuracies2[i] - accuracies1[i] for i in range(3)]
colors = ['green' if x > 0 else 'red' for x in improvements]
axes[2].bar(CLASS_NAMES, improvements, color=colors)
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_ylabel('Improvement (%)')
axes[2].set_title('Accuracy Improvement')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nComparison plot saved: model_comparison.png")