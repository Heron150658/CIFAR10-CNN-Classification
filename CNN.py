import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten
)
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import seaborn as sns

# =====================================================
# 参数设置
# =====================================================

BATCH_SIZE = 64
EPOCHS = 5
NUM_CLASSES = 10

CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

# =====================================================
# 加载数据
# =====================================================

print("Loading CIFAR-10 dataset...")

(x_train, y_train_raw), (x_test, y_test_raw) = cifar10.load_data()

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# =====================================================
# 数据可视化1：样本展示
# =====================================================

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(CLASS_NAMES[y_train_raw[i][0]])
    plt.axis("off")

plt.suptitle("CIFAR-10 Sample Images")
plt.tight_layout()
plt.show()

# =====================================================
# 数据可视化2：类别分布
# =====================================================

plt.figure(figsize=(8, 5))

labels = y_train_raw.flatten()

plt.hist(labels, bins=10)

plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution")

plt.show()

# =====================================================
# 数据预处理
# =====================================================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = to_categorical(y_train_raw, NUM_CLASSES)
y_test = to_categorical(y_test_raw, NUM_CLASSES)

print("y_train shape:", y_train.shape)

# =====================================================
# 构建CNN
# =====================================================

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        input_shape=(32, 32, 3)
    )
)

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu"
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

model.add(
    Dropout(0.25)
)

model.add(
    Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )
)

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

model.add(
    Dropout(0.25)
)

model.add(Flatten())

model.add(
    Dense(
        512,
        activation="relu"
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
)

# =====================================================
# 编译模型
# =====================================================

optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=0.001
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

# =====================================================
# 训练模型
# =====================================================

print("\nTraining...\n")

history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    shuffle=True
)

# =====================================================
# 保存模型
# =====================================================

save_dir = "saved_models"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(
    save_dir,
    "keras_cifar10_trained_model.h5"
)

model.save(model_path)

print("\nModel saved to:")
print(model_path)

# =====================================================
# 测试集评估
# =====================================================

scores = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\nTest Loss:", scores[0])
print("Test Accuracy:", scores[1])

# =====================================================
# Accuracy曲线
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Curve")

plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# Loss曲线
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# 分类报告
# =====================================================

y_pred_prob = model.predict(x_test)

y_pred = np.argmax(
    y_pred_prob,
    axis=1
)

y_true = np.argmax(
    y_test,
    axis=1
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    )
)

# =====================================================
# 混淆矩阵
# =====================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.show()

# =====================================================
# 正确分类案例
# =====================================================

correct_idx = np.where(
    y_pred == y_true
)[0]

plt.figure(figsize=(10, 5))

for i in range(5):

    idx = correct_idx[i]

    plt.subplot(1, 5, i + 1)

    plt.imshow(x_test[idx])

    plt.title(
        CLASS_NAMES[y_pred[idx]]
    )

    plt.axis("off")

plt.suptitle("Correct Predictions")

plt.show()

# =====================================================
# 错误分类案例
# =====================================================

wrong_idx = np.where(
    y_pred != y_true
)[0]

plt.figure(figsize=(12, 6))

for i in range(5):

    idx = wrong_idx[i]

    plt.subplot(1, 5, i + 1)

    plt.imshow(x_test[idx])

    plt.title(
        f"T:{CLASS_NAMES[y_true[idx]]}\nP:{CLASS_NAMES[y_pred[idx]]}"
    )

    plt.axis("off")

plt.suptitle("Wrong Predictions")

plt.show()





