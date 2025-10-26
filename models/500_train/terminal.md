root@cefeffefda2c:/# python train.py
2025-10-26 16:18:01.445143: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
============================================================
🎯 CLASIFICADOR CIFAR-10
============================================================
Clases a entrenar: 3
Muestras de entrenamiento: 500
Épocas: 30
Batch size: 2
Learning rate: 3e-4
============================================================

📥 Cargando CIFAR-10...
Filtrando clases: ['airplane', 'automobile', 'ship']
⚠️  Limitando dataset a 500 muestras de entrenamiento
   (de 15000 disponibles)
✅ Datos preparados:
   Train: 500 muestras
   Test: 3000 muestras
   Clases: ['airplane', 'automobile', 'ship']

📊 Split de datos:
   Train: 400
   Validation: 100
   Test: 3000

🧠 Creando modelo...
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1761495485.902581   42445 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22322 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:82:00.0, compute capability: 8.6
✅ Modelo creado con 3 clases
Model: "CIFAR10_Classifier"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ resizing (Resizing)                  │ (None, 224, 224, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ MobileNetV3Small (Functional)        │ (None, 7, 7, 576)           │         939,120 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling2d             │ (None, 576)                 │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 576)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 3)                   │           1,731 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 940,851 (3.59 MB)
 Trainable params: 928,739 (3.54 MB)
 Non-trainable params: 12,112 (47.31 KB)
🚀 Iniciando entrenamiento...
============================================================

Epoch 1/30
2025-10-26 16:18:18.473683: I external/local_xla/xla/service/service.cc:163] XLA service 0x771b680032d0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2025-10-26 16:18:18.473711: I external/local_xla/xla/service/service.cc:171]   StreamExecutor device (0): NVIDIA GeForce RTX 3090, Compute Capability 8.6
2025-10-26 16:18:18.803319: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2025-10-26 16:18:20.498107: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:473] Loaded cuDNN version 91002
I0000 00:00:1761495515.268998   42833 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
199/200 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.3873 - loss: 1.2672
Epoch 1: val_accuracy improved from None to 0.32000, saving model to models/best_model.keras
200/200 ━━━━━━━━━━━━━━━━━━━━ 35s 33ms/step - accuracy: 0.4675 - loss: 1.1026 - val_accuracy: 0.3200 - val_loss: 1.3859
Epoch 2/30
195/200 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.6113 - loss: 0.8001
Epoch 2: val_accuracy did not improve from 0.32000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 12ms/step - accuracy: 0.6575 - loss: 0.7793 - val_accuracy: 0.3200 - val_loss: 1.1776
Epoch 3/30
196/200 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.8140 - loss: 0.5222
Epoch 3: val_accuracy did not improve from 0.32000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 11ms/step - accuracy: 0.7925 - loss: 0.5577 - val_accuracy: 0.3200 - val_loss: 1.3128
Epoch 4/30
197/200 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.8104 - loss: 0.4575
Epoch 4: val_accuracy did not improve from 0.32000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.8225 - loss: 0.4505 - val_accuracy: 0.3200 - val_loss: 1.1834
Epoch 5/30
197/200 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.8611 - loss: 0.4062
Epoch 5: val_accuracy did not improve from 0.32000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 12ms/step - accuracy: 0.8550 - loss: 0.3979 - val_accuracy: 0.3200 - val_loss: 1.1089
Epoch 6/30
200/200 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.9316 - loss: 0.2470
Epoch 6: val_accuracy improved from 0.32000 to 0.34000, saving model to models/best_model.keras
200/200 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.9300 - loss: 0.2396 - val_accuracy: 0.3400 - val_loss: 1.1908
Epoch 7/30
195/200 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9248 - loss: 0.1992
Epoch 7: val_accuracy did not improve from 0.34000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9075 - loss: 0.2370 - val_accuracy: 0.3400 - val_loss: 1.3192
Epoch 8/30
199/200 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9622 - loss: 0.1425
Epoch 8: val_accuracy did not improve from 0.34000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9550 - loss: 0.1528 - val_accuracy: 0.3400 - val_loss: 1.2579
Epoch 9/30
197/200 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9916 - loss: 0.0706
Epoch 9: val_accuracy did not improve from 0.34000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9725 - loss: 0.1019 - val_accuracy: 0.3200 - val_loss: 1.4088
Epoch 10/30
200/200 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9427 - loss: 0.1523
Epoch 10: val_accuracy did not improve from 0.34000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9550 - loss: 0.1333 - val_accuracy: 0.2700 - val_loss: 1.3641
Epoch 11/30
200/200 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9889 - loss: 0.0557
Epoch 11: val_accuracy did not improve from 0.34000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9850 - loss: 0.0833 - val_accuracy: 0.3200 - val_loss: 1.4910
Epoch 12/30
199/200 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.9974 - loss: 0.0435
Epoch 12: val_accuracy did not improve from 0.34000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 11ms/step - accuracy: 0.9900 - loss: 0.0486 - val_accuracy: 0.3400 - val_loss: 1.5105
Epoch 13/30
200/200 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.9878 - loss: 0.0389
Epoch 13: val_accuracy did not improve from 0.34000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9850 - loss: 0.0462 - val_accuracy: 0.3400 - val_loss: 1.7136
Epoch 14/30
199/200 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9921 - loss: 0.0427
Epoch 14: val_accuracy did not improve from 0.34000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9850 - loss: 0.0500 - val_accuracy: 0.3200 - val_loss: 3.1044
Epoch 15/30
199/200 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9750 - loss: 0.0799
Epoch 15: val_accuracy did not improve from 0.34000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9825 - loss: 0.0623 - val_accuracy: 0.3200 - val_loss: 3.1285
Epoch 16/30
197/200 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9918 - loss: 0.0354
Epoch 16: val_accuracy did not improve from 0.34000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9850 - loss: 0.0461 - val_accuracy: 0.3100 - val_loss: 1.5831
Epoch 17/30
195/200 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9857 - loss: 0.0514
Epoch 17: val_accuracy improved from 0.34000 to 0.35000, saving model to models/best_model.keras
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 12ms/step - accuracy: 0.9775 - loss: 0.0538 - val_accuracy: 0.3500 - val_loss: 1.2989
Epoch 18/30
194/200 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.9995 - loss: 0.0239
Epoch 18: val_accuracy did not improve from 0.35000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9975 - loss: 0.0197 - val_accuracy: 0.3400 - val_loss: 1.3721
Epoch 19/30
199/200 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.9796 - loss: 0.0544
Epoch 19: val_accuracy improved from 0.35000 to 0.38000, saving model to models/best_model.keras
200/200 ━━━━━━━━━━━━━━━━━━━━ 3s 12ms/step - accuracy: 0.9725 - loss: 0.0706 - val_accuracy: 0.3800 - val_loss: 1.2015
Epoch 20/30
192/200 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9965 - loss: 0.0263
Epoch 20: val_accuracy improved from 0.38000 to 0.44000, saving model to models/best_model.keras
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 11ms/step - accuracy: 0.9900 - loss: 0.0359 - val_accuracy: 0.4400 - val_loss: 1.0586
Epoch 21/30
197/200 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 1.0000 - loss: 0.0360
Epoch 21: val_accuracy did not improve from 0.44000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 1.0000 - loss: 0.0196 - val_accuracy: 0.4000 - val_loss: 1.0843
Epoch 22/30
194/200 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9697 - loss: 0.0846
Epoch 22: val_accuracy did not improve from 0.44000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9775 - loss: 0.0608 - val_accuracy: 0.4400 - val_loss: 1.2383
Epoch 23/30
198/200 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9888 - loss: 0.0332
Epoch 23: val_accuracy improved from 0.44000 to 0.50000, saving model to models/best_model.keras
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 11ms/step - accuracy: 0.9850 - loss: 0.0454 - val_accuracy: 0.5000 - val_loss: 1.4684
Epoch 24/30
194/200 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9826 - loss: 0.0476
Epoch 24: val_accuracy improved from 0.50000 to 0.65000, saving model to models/best_model.keras
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9875 - loss: 0.0343 - val_accuracy: 0.6500 - val_loss: 0.9497
Epoch 25/30
197/200 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9958 - loss: 0.0212
Epoch 25: val_accuracy improved from 0.65000 to 0.74000, saving model to models/best_model.keras
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9950 - loss: 0.0244 - val_accuracy: 0.7400 - val_loss: 0.7207
Epoch 26/30
198/200 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9902 - loss: 0.0255
Epoch 26: val_accuracy did not improve from 0.74000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9925 - loss: 0.0211 - val_accuracy: 0.7300 - val_loss: 0.7001
Epoch 27/30
194/200 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9983 - loss: 0.0109
Epoch 27: val_accuracy did not improve from 0.74000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9925 - loss: 0.0263 - val_accuracy: 0.3100 - val_loss: 2.2114
Epoch 28/30
195/200 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 1.0000 - loss: 0.0229
Epoch 28: val_accuracy did not improve from 0.74000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 1.0000 - loss: 0.0122 - val_accuracy: 0.6700 - val_loss: 1.6959
Epoch 29/30
200/200 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9904 - loss: 0.0109
Epoch 29: val_accuracy did not improve from 0.74000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9950 - loss: 0.0078 - val_accuracy: 0.6900 - val_loss: 1.2397
Epoch 30/30
196/200 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 1.0000 - loss: 0.0036
Epoch 30: val_accuracy did not improve from 0.74000
200/200 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 1.0000 - loss: 0.0034 - val_accuracy: 0.6700 - val_loss: 1.1772

============================================================
📊 EVALUACIÓN FINAL
============================================================
2025-10-26 16:19:43.848661: I external/local_xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_1430', 128 bytes spill stores, 128 bytes spill loads

2025-10-26 16:19:43.867648: I external/local_xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_1430', 172 bytes spill stores, 172 bytes spill loads

2025-10-26 16:19:50.651828: I external/local_xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_1430', 132 bytes spill stores, 132 bytes spill loads

2025-10-26 16:19:50.716216: I external/local_xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_1430', 136 bytes spill stores, 136 bytes spill loads

🎯 Test Loss: 0.9885
🎯 Test Accuracy: 0.6687 (66.87%)

✅ Modelo guardado en: models/final_model.keras
📈 Generando gráficas...
✅ Gráfica guardada en: models/training_history.png

============================================================
🎉 ENTRENAMIENTO COMPLETADO
============================================================

📋 Archivos generados:
   - models/best_model.keras
   - models/final_model.keras
   - models/training_history.png

🔄 Siguiente paso: Implementar Active Learning