from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, TimeDistributed
from tensorflow.keras.layers import LSTM
import os

os.makedirs("models", exist_ok=True)

# ================= IMAGE MODEL =================
image_model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(224,224,3)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

image_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

image_model.save("models/image_model.h5")
print("✅ image_model.h5 created")

# ================= VIDEO MODEL =================
video_model = Sequential([
    TimeDistributed(
        Conv2D(8, (3,3), activation='relu'),
        input_shape=(None, 224, 224, 3)
    ),
    TimeDistributed(Flatten()),
    LSTM(16),
    Dense(1, activation='sigmoid')
])

video_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

video_model.save("models/video_model.h5")
print("✅ video_model.h5 created")