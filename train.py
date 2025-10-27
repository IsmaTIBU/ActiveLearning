import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


NUM_CLASSES = 3
SELECTED_CLASSES = [0, 1, 8]
TRAIN_SAMPLES = 500
NUM_EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 3e-4


def load_and_prepare_data(num_classes=3, selected_classes=None, train_samples=None, 
                         additional_indices=None):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    if selected_classes is not None:
        mask_train = np.isin(y_train.flatten(), selected_classes)
        x_train = x_train[mask_train]
        y_train = y_train[mask_train]
        
        mask_test = np.isin(y_test.flatten(), selected_classes)
        x_test = x_test[mask_test]
        y_test = y_test[mask_test]
        
        for new_idx, old_idx in enumerate(selected_classes):
            y_train[y_train == old_idx] = new_idx
            y_test[y_test == old_idx] = new_idx
    
    if train_samples is not None and train_samples < len(x_train):
        indices = np.random.permutation(len(x_train))
        initial_indices = indices[:train_samples]
        
        if additional_indices is not None:
            combined_indices = np.concatenate([initial_indices, additional_indices])
            combined_indices = np.unique(combined_indices)
        else:
            combined_indices = initial_indices
        
        x_train = x_train[combined_indices]
        y_train = y_train[combined_indices]
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return (x_train, y_train), (x_test, y_test)


def create_model(num_classes=3):
    base_model = keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True
    
    model = keras.Sequential([
        keras.layers.Input(shape=(32, 32, 3)),
        keras.layers.Resizing(224, 224),
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ], name='CIFAR10_Classifier')
    
    return model


def plot_history(history, output_dir='models'):
    """Grafica el historial de entrenamiento"""
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', linewidth=2, label='Training')
    plt.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation')
    plt.title('Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', linewidth=2, label='Training')
    plt.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation')
    plt.title('Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def train_model(additional_indices=None, output_dir='models', verbose=True):
    """
    Entrena el modelo
    
    Args:
        additional_indices: Índices adicionales desde intersection (None = standalone)
        output_dir: Directorio donde guardar el modelo
        verbose: Mostrar información de entrenamiento
    
    Returns:
        model: Modelo entrenado
        history: Historial de entrenamiento
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data(
        num_classes=NUM_CLASSES,
        selected_classes=SELECTED_CLASSES,
        train_samples=TRAIN_SAMPLES,
        additional_indices=additional_indices
    )
    
    if verbose:
        print(f"Training with {len(x_train)} images")
        if additional_indices is not None:
            print(f"  Initial: {TRAIN_SAMPLES}")
            print(f"  Additional: {len(additional_indices)}")
    
    val_split = 0.2
    val_size = int(len(x_train) * val_split)
    
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train_subset = x_train[:-val_size]
    y_train_subset = y_train[:-val_size]
    
    model = create_model(num_classes=NUM_CLASSES)
    
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f'{output_dir}/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )
    ]
    
    history = model.fit(
        x_train_subset, y_train_subset,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1 if verbose else 0
    )
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    if verbose:
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    plot_history(history, output_dir)
    
    if verbose:
        print(f"Training plot saved: {output_dir}/training_history.png")
    
    return model, history


if __name__ == "__main__":
    print("="*60)
    print("STANDALONE TRAINING - 500 images")
    print("="*60)
    
    model, history = train_model(additional_indices=None, verbose=True)
    
    print("\nTraining completed!")
    print("Model saved: models/best_model.keras")