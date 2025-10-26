"""
Entrenamiento CIFAR-10 con TensorFlow/Keras
Basado en MobileNetV3 para clasificación multi-clase
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ========================================
# CONFIGURACIÓN - CAMBIA AQUÍ
# ========================================
NUM_CLASSES = 3  # Cambia a 10 para todas las clases
SELECTED_CLASSES = [0, 1, 8]  # airplane, automobile, ship (o None para todas)
TRAIN_SAMPLES = 500  # Número de imágenes para entrenamiento inicial (None = todas)
NUM_EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 3e-4


print("=" * 60)
print("🎯 CLASIFICADOR CIFAR-10")
print("=" * 60)
print(f"Clases a entrenar: {NUM_CLASSES}")
print(f"Muestras de entrenamiento: {TRAIN_SAMPLES if TRAIN_SAMPLES else 'TODAS'}")
print(f"Épocas: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print("=" * 60)
print()


# ========================================
# 1. CARGAR Y PREPARAR DATOS
# ========================================
def load_and_prepare_data(num_classes=3, selected_classes=None, train_samples=None):
    """
    Carga y prepara CIFAR-10
    
    Args:
        num_classes: Número de clases (3 o 10)
        selected_classes: Lista de índices de clases [0,1,8] o None para todas
        train_samples: Número de muestras de entrenamiento a usar (None = todas)
    """
    
    print("📥 Cargando CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Filtrar clases si se especifica
    if selected_classes is not None:
        print(f"Filtrando clases: {[class_names[i] for i in selected_classes]}")
        
        mask_train = np.isin(y_train.flatten(), selected_classes)
        x_train = x_train[mask_train]
        y_train = y_train[mask_train]
        
        mask_test = np.isin(y_test.flatten(), selected_classes)
        x_test = x_test[mask_test]
        y_test = y_test[mask_test]
        
        # Remapear labels a 0, 1, 2, ...
        for new_idx, old_idx in enumerate(selected_classes):
            y_train[y_train == old_idx] = new_idx
            y_test[y_test == old_idx] = new_idx
        
        selected_names = [class_names[i] for i in selected_classes]
    else:
        selected_names = class_names
    
    # LIMITAR número de muestras de entrenamiento si se especifica
    if train_samples is not None and train_samples < len(x_train):
        print(f"⚠️  Limitando dataset a {train_samples} muestras de entrenamiento")
        
        # Mezclar aleatoriamente para tener variedad
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices[:train_samples]]
        y_train = y_train[indices[:train_samples]]
        
        print(f"   (de {len(indices)} disponibles)")
    
    # Normalizar
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # One-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    print(f"✅ Datos preparados:")
    print(f"   Train: {x_train.shape[0]} muestras")
    print(f"   Test: {x_test.shape[0]} muestras")
    print(f"   Clases: {selected_names}")
    print()
    
    return (x_train, y_train), (x_test, y_test), selected_names


# ========================================
# 2. CREAR MODELO
# ========================================
def create_model(num_classes=3):
    """
    Crea modelo basado en MobileNetV3Small
    
    Args:
        num_classes: Número de clases de salida
    """
    
    print("🧠 Creando modelo...")
    
    # Base model: MobileNetV3Small (ligero y eficiente)
    base_model = keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Descongelar para fine-tuning
    base_model.trainable = True
    
    # Construir modelo completo
    model = keras.Sequential([
        keras.layers.Input(shape=(32, 32, 3)),
        keras.layers.Resizing(224, 224),  # CIFAR-10 es 32x32, MobileNet necesita 224x224
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ], name='CIFAR10_Classifier')
    
    print(f"✅ Modelo creado con {num_classes} clases")
    
    # Mostrar resumen
    model.summary()
    
    return model, base_model


# ========================================
# 3. ENTRENAMIENTO
# ========================================
def train_model():
    """Entrena el modelo"""
    
    # Crear directorio para modelos
    Path('models').mkdir(exist_ok=True)
    
    # Cargar datos
    (x_train, y_train), (x_test, y_test), class_names = load_and_prepare_data(
        num_classes=NUM_CLASSES,
        selected_classes=SELECTED_CLASSES,
        train_samples=TRAIN_SAMPLES  # Limitar número de muestras
    )
    
    # Split train/validation
    val_split = 0.2
    val_size = int(len(x_train) * val_split)
    
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train_subset = x_train[:-val_size]
    y_train_subset = y_train[:-val_size]
    
    print(f"📊 Split de datos:")
    print(f"   Train: {len(x_train_subset)}")
    print(f"   Validation: {len(x_val)}")
    print(f"   Test: {len(x_test)}")
    print()
    
    # Crear modelo
    model, base_model = create_model(num_classes=NUM_CLASSES)
    
    # Compilar
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("🚀 Iniciando entrenamiento...")
    print("=" * 60)
    print()
    
    # Entrenar
    history = model.fit(
        x_train_subset, y_train_subset,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluación final
    print()
    print("=" * 60)
    print("📊 EVALUACIÓN FINAL")
    print("=" * 60)
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"🎯 Test Loss: {test_loss:.4f}")
    print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print()
    
    # Guardar modelo final
    model.save('models/final_model.keras')
    print("✅ Modelo guardado en: models/final_model.keras")
    
    # Graficar historial
    plot_history(history)
    
    return model, history


# ========================================
# 4. VISUALIZACIÓN
# ========================================
def plot_history(history):
    """Grafica el historial de entrenamiento"""
    
    print("📈 Generando gráficas...")
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', linewidth=2, label='Training')
    plt.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation')
    plt.title('Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', linewidth=2, label='Training')
    plt.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation')
    plt.title('Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
    print("✅ Gráfica guardada en: models/training_history.png")
    plt.show()


# ========================================
# 5. EJECUTAR
# ========================================
if __name__ == "__main__":
    model, history = train_model()
    
    print()
    print("=" * 60)
    print("🎉 ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print()
    print("📋 Archivos generados:")
    print("   - models/best_model.keras")
    print("   - models/final_model.keras")
    print("   - models/training_history.png")
    print()
    print("🔄 Siguiente paso: Implementar Active Learning")