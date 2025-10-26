"""
Descarga CIFAR-10 con barra de progreso
"""

import tensorflow as tf
from tensorflow import keras
import sys
from pathlib import Path


def download_cifar10():
    """Descarga CIFAR-10 mostrando progreso"""
    
    print("=" * 60)
    print("📥 DESCARGANDO CIFAR-10 DATASET")
    print("=" * 60)
    print()
    
    # Directorio donde se guardará
    data_dir = Path('dataset/cifar10')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Guardando en: {data_dir.absolute()}")
    print()
    
    try:
        print("⏳ Descargando... (esto puede tardar 1-2 minutos)")
        print()
        
        # Keras descarga automáticamente con barra de progreso integrada
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        print()
        print("=" * 60)
        print("✅ DESCARGA COMPLETADA")
        print("=" * 60)
        print()
        print(f"📊 Dataset Info:")
        print(f"   Train images: {x_train.shape[0]:,}")
        print(f"   Test images:  {x_test.shape[0]:,}")
        print(f"   Image size:   {x_train.shape[1]}x{x_train.shape[2]}")
        print(f"   Channels:     {x_train.shape[3]} (RGB)")
        print()
        print("📋 Clases:")
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        for i, name in enumerate(class_names):
            print(f"   {i}: {name}")
        print()
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ ERROR EN LA DESCARGA")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("💡 Sugerencias:")
        print("   - Verifica tu conexión a internet")
        print("   - Intenta de nuevo en unos minutos")
        return False


if __name__ == "__main__":
    success = download_cifar10()
    
    if success:
        print("🎉 Listo para entrenar!")
        print("▶️  Ejecuta: python train.py")
    else:
        print("⚠️  Por favor, soluciona el error e intenta de nuevo")