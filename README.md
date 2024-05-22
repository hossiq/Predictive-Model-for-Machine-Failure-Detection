# Predictive-Model-for-Machine-Failure-Detection

**Background**

Fourth Industrial Revolution (Industry 4.0) represents the current trend of automation and data exchange in manufacturing and other industries, integrating cyber-physical systems, the Internet of Things (IoT), cloud computing, big data, and cognitive computing.Smart monitoring of manufacturing assets plays a critical role in enhancing operational efficiency and optimizing key performance indicators such as Mean Time Between Failures (MTBF), Mean Time to Repair (MTTR), and Overall Equipment Effectiveness (OEE).The proactive detection of machine failure is essential to mitigating downtime costs and sustaining manufacturing productivity

**Objective**

The project proposes an intelligent monitoring system that employs advanced machine learning techniques to analyze sensor data from the machines, Which eventually can be applied for Reduction in Unscheduled Downtime and Optimization of Maintenance Schedules.

**Data Collection, Analysis, and Processing**

A Water Pump data is collected from the Kaggle website https://www.kaggle.com/datasets/nphantawee/pump-sensor-data for this study. This study used three months of data April, May, and June of 2018. Dataset contains 52 sensors , Timestamp by Minute, and Machine Status. I replace columns with nulls with mean value and dropped duplicates.
<p align="center">
  <img src="https://github.com/hossiq/image/blob/main/Machine Status.png?raw=true" alt="Plot" width="450"/>
</p>

It's noted from above picture only 0.0038% data belongs to 'BROKEN' status which we are going to predict.

**Long Short-Term Memory (LSTM)**

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) capable of learning order dependence in sequence prediction problems. It can effectively handle the temporal variability in sensor data, making it ideal for predicting equipment failures. To use LSTM, sensor data is normalized for better training performance; Transformed time-series sensor data into sequences to serve as input for the LSTM, ensuring each sequence captures enough historical context, with sequence length 30 represents 30 minutes of data; Two LSTM layers used with regularization to prevent overfitting and improve model generalization; Applied dropout layers between LSTM layers to further mitigate the risk of overfitting; Trained the model over 5 epochs with a batch size of 8 to balance speed and memory usage; and Monitored Accuracy, Precision, Recall, AUC metrics to evaluate the model’s ability to accurately predict machine status.

 <pre>
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time

# Step 1: Load Data
file_path = '/content/sample_data/Sensor Data/sensor.csv'
data = pd.read_csv(file_path, parse_dates=['timestamp'])
print("Data loaded.")

# Step 2: Preprocess Data
data = data[(data['timestamp'].dt.month == 4) | (data['timestamp'].dt.month == 5)| (data['timestamp'].dt.month == 6)]


# Remove column 15 ('sensor_15') because this column is blank
data.drop('sensor_15', axis=1, inplace=True)

# Replace columns with nulls with mean value
sensor_columns = data.columns.difference(['timestamp', 'machine_status'])
data[sensor_columns] = data[sensor_columns].apply(lambda x: x.fillna(x.mean()), axis=0)

# Drop any duplicates
data.drop_duplicates(inplace=True)

# Encode machine status column
conditions = [
    (data['machine_status'] == 'NORMAL'),
    (data['machine_status'] == 'BROKEN'),
    (data['machine_status'] == 'RECOVERING')
]
choices = [1, 0, 2]
data['machine_status'] = np.select(conditions, choices)
print(data['machine_status'].value_counts())

# Normalize features
scaler = MinMaxScaler()
data[sensor_columns] = scaler.fit_transform(data[sensor_columns])

# Step 3: Create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in tqdm(range(len(data) - sequence_length), desc='Creating sequences'):
        sequence = data.iloc[i:i+sequence_length][sensor_columns].values
        label = data.iloc[i+sequence_length]['machine_status']
        X.append(sequence)
        y.append(label)
    return np.array(X), np.array(y)

sequence_length = 30  # Simplified to 30 minutes worth of data
X, y = create_sequences(data, sequence_length)
print("Sequences created.")

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print("Dataset split into train and test sets.")

# Step 5: Build the model
model = Sequential()
model.add(LSTM(20, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(LSTM(20, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall(), AUC()])
print("Model built.")

# Step 6: Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=8, validation_split=0.1, verbose=1)
print("Model trained.")

# Step 7: Evaluate the model
evaluation = model.evaluate(X_test, y_test, verbose=1)
print("Model evaluated.")
print(f"Model Evaluation:\n Loss: {evaluation[0]}\n Accuracy: {evaluation[1]}\n Precision: {evaluation[2]}\n Recall: {evaluation[3]}\n AUC: {evaluation[4]}")

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

  ```
</pre>





