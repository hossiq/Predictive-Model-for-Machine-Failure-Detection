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


**LSTM Model Output**



<img src="https://github.com/hossiq/image/blob/main/LSTM_Out.PNG?raw=true" alt="rf" style="width: 350px; object-fit: cover;"/><img src="https://github.com/hossiq/image/blob/main/LSTM_Out_2.PNG?raw=true"  alt="smote" style="width: 350px; object-fit: cover;"/>

The loss function is low (0.0258) which quantifies the difference between the predicted and actual values and the output indicates a close match between predictions and actual machine statuses. Model shows High Accuracy, Precision, and Recall. Area Under Curve (AUC) = 0.5 indicates that the model is not distinguishing between the positive and negative classes. Usually, an AUC of 0.7 to 0.8 is considered acceptable, 0.8 to 0.9 is considered excellent, and above 0.9 is considered outstanding.

Despite achieving perfect accuracy and precision, the model's AUC of 0.5 suggests it fails to effectively predict the critical 'BROKEN' category (only 0.0038% data), indicating a need for further model refinement and data rebalancing to enhance predictive performance for equipment failures


**Model Development: Anomaly Detection with Autoencoder**

After observing suboptimal performance with LSTM model, indicated by an AUC of 0.5, this project explored alternative approach (Autoencoder) to improve our predictive maintenance capabilities.
Autoencoder is a type of neural network used to learn unlabeled data. It works by compressing the input into a latent-space representation and then reconstructing the output from this representation. It particularly suitable at capturing the normal state of the system. The model architecture is streamlined to have a single hidden layer with 14 neurons, chosen to balance model complexity and performance.To prevent overfitting, early stopping is used. And Established threshold at the 95th percentile and the 99th percentile of the normal data's reconstruction error as a criterion for detecting anomalies. Reconstruction error is a measure of the difference between the original input data and the output produced by the autoencoder.

 <pre>
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Load data
file_path = '/content/sample_data/Sensor Data/sensor.csv'
data = pd.read_csv(file_path, parse_dates=['timestamp'])
data = data[(data['timestamp'].dt.month == 4) | (data['timestamp'].dt.month == 5)| (data['timestamp'].dt.month == 6)]

# Remove column 'sensor_15'
data.drop('sensor_15', axis=1, inplace=True)

# Split data into normal and broken
normal_data = data[data['machine_status'] == 'NORMAL'].copy()
broken_data = data[data['machine_status'] == 'BROKEN'].copy()

# Drop unnecessary columns
normal_data.drop(['timestamp', 'machine_status'], axis=1, inplace=True)
broken_data.drop(['timestamp', 'machine_status'], axis=1, inplace=True)

# Normalize features for normal data
scaler = MinMaxScaler()
normal_scaled = scaler.fit_transform(normal_data.fillna(normal_data.mean()))

# Split normal data into training and test sets
X_train, X_test = train_test_split(normal_scaled, test_size=0.2, random_state=42)

# Define autoencoder model architecture
input_dim = X_train.shape[1]
encoding_dim = 14  # You might consider tuning this

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

# Checkpoint callback to save the best model
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=True)

# Fit the model with training data
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_split=0.2,
                verbose=0,
                callbacks=[TqdmCallback(verbose=2), early_stopping, checkpoint])

# Load the best weights
autoencoder.load_weights('best_model.h5')

# Evaluate the model on the test set
test_mse = np.mean(np.power(X_test - autoencoder.predict(X_test), 2), axis=1)

# Prepare the broken data
broken_scaled = scaler.transform(broken_data.fillna(broken_data.mean()))

# Evaluate the model on the broken set
broken_mse = np.mean(np.power(broken_scaled - autoencoder.predict(broken_scaled), 2), axis=1)

# Determine the threshold as the 99th percentile of the normal test set MSE
threshold = np.quantile(test_mse, 0.95)

# Identify anomalies (where the MSE of the reconstruction is greater than the threshold)
anomalies = broken_mse > threshold

# Plot histograms
plt.hist(test_mse, bins=50, alpha=0.5, color='blue', label='Normal')
plt.hist(broken_mse, bins=50, alpha=0.5, color='red', label='Broken')
plt.axvline(threshold, color='green', linestyle='dashed', linewidth=2, label='Threshold')
plt.title('Histogram of Mean Squared Errors')
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Print results
print(f'Reconstruction error threshold: {threshold}')
print(f'Detected anomalies out of broken instances: {np.sum(anomalies)} / {len(broken_mse)}')
 ```
</pre>

**Autoencoder Model Output**

<img src="https://github.com/hossiq/image/blob/main/Anomaly_95th.png?raw=true"  alt="smote" style="width: 280px; object-fit: cover;"/><img src="https://github.com/hossiq/image/blob/main/Anomaly_99th.png?raw=true"  alt="ADASYN" style="width: 280px; object-fit: cover;"/>

<img src="https://github.com/hossiq/image/blob/main/Autoencoder.png?raw=true" alt="rf" style="width: 280px; object-fit: cover;"/>


The model's sensitivity to anomalies increases with a lower reconstruction error threshold, as demonstrated by the 100% detection rate at the 95th percentile (0.0034). At the 99th percentile (0.008), the model exhibits a decrease in sensitivity, detecting 60% of known anomalies. 

This suggests a trade-off, accepting fewer detections for the benefit of reducing false positives and focusing on more significant deviations from the norm. A lower threshold (95th percentile) is suited for high-alert environments, while a higher threshold (99th percentile) may be better for systems where false alarms are particularly disruptive.



**Conclusion**


The results shows the importance of threshold tuning in anomaly detection models and highlight the potential need for a hybrid approach that incorporates the sensitivity of Autoencoders with the specificity of other models. Author suggests that, creating an ensemble that leverages the strengths of both LSTM and Autoencoder methodologies would provide better, consistent output




