import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pandas as pd
import os
import collect_train
import train_eval
import pickle as pkl

ct = collect_train.collect_train()

# Get desired model	
def get_model(algo):
	pklfile = open('../pkl/{}pickle.pkl'.format(algo), 'rb')  
	model = pkl.load(pklfile)
	pklfile.close()
	return model

# Get the model
hgb = get_model('hgb')

# Infinite loop
while(True):
	res, frame = ct.cap.read()
	if not res:
		print("Couldn't grab frame, continuing to next iteration")
	else:
		# Convert to RGB (opencv is natively BGR)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# Get hand landmarks 
		results = ct.hands.process(frame)
		# Show landmarks superimposed on hand
		stat, image = ct.drawHands(results, frame)
		if stat != 0:
			print("Didn't detect hand")
			continue
		else:
			tab = ct.tabulate_test(results)
			pred = hgb.predict(tab)
			print("Prediction: {}".format(pred[0]))
			image = cv2.flip(image, 1)
			image = cv2.putText(image, pred[0], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
			cv2.imshow("Landmarks", image)
			if cv2.waitKey(5) & 0xFF == 27:
				# Exit demo
				break

cv2.destroyAllWindows()