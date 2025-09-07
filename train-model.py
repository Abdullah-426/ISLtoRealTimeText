from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os
import shutil


# ----------------- Data Augmentation -----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 16  # small dataset → small batch size

train_generator = train_datagen.flow_from_directory(
    'smaller-data/train',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    'smaller-data/test',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

class_names = list(train_generator.class_indices.keys())
print("Classes:", class_names)

num_classes = len(class_names)  # should be 35

# ----------------- Model (Simpler CNN) -----------------
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# ----------------- Compile -----------------
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# ----------------- TensorBoard -----------------
if os.path.exists("Logs"):
    shutil.rmtree("Logs")
logdir = "Logs"
tensorboard_callback = TensorBoard(log_dir=logdir)

# ----------------- Early Stopping -----------------
earlystop_callback = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

# ----------------- Training -----------------
model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // batch_size),
    epochs=20,   # reduced for small dataset
    validation_data=test_generator,
    validation_steps=max(1, test_generator.samples // batch_size),
    callbacks=[tensorboard_callback, earlystop_callback]
)

# ----------------- Save Model -----------------
model_json = model.to_json()
with open("sign_model.json", "w") as json_file:
    json_file.write(model_json)
model.save("sign_model.h5")
