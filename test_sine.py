import os
import json
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model

configs = json.load(open('config.json', 'r'))

data = DataLoader(
	os.path.join('data', configs['data']['filename']),
	configs['data']['train_test_split'],
	configs['data']['columns']
)

model = Model()
model.build_model(configs)
x, y = data.get_train_data(
	seq_len = configs['data']['sequence_length'],
	normalise = configs['data']['normalise']
)

model.train(
	x,
	y,
	epochs = configs['training']['epochs'],
	batch_size = configs['training']['batch_size'],
    save_dir = configs['model']['save_dir']
)

x_test, y_test = data.get_test_data(
	seq_len = configs['data']['sequence_length'],
	normalise = configs['data']['normalise']
)

def predict_point_by_point(self, data):
    	#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
	predicted = self.model.predict(data)
	predicted = np.reshape(predicted, (predicted.size,))
	return predicted

def predict_sequence_full(self, data, window_size):
	#Shift the window by 1 new prediction each time, re-run predictions on new window
	curr_frame = data[0]
	predicted = []
	for i in range(len(data)):
		predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
		curr_frame = curr_frame[1:]
		curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
	return predicted

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

predictions_pointbypoint = model.predict_point_by_point(x_test)
plot_results(predictions_pointbypoint, y_test)

predictions_fullseq = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
plot_results(predictions_fullseq, y_test)