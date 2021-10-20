import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pandas as pd
import os

class collect_train:

	#constructor
	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		# setup google hands api
		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_drawing_styles = mp.solutions.drawing_styles
		self.DEMO_LENGTH = 10000
		self.overall_df = pd.DataFrame()
	
	#destructor
	def __del__(self):
		self.cap.release()

	# Helper function for drawing the landmarks on the screen
	def drawHands(self, results, image):
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				self.mp_drawing.draw_landmarks(
					image,
					hand_landmarks,
					self.mp_hands.HAND_CONNECTIONS,
					self.mp_drawing_styles.get_default_hand_landmarks_style(),
					self.mp_drawing_styles.get_default_hand_connections_style())

			retval = 0
		else:
			retval = -1

		return retval, image


	# Convert landmarks (and class) to tabular form
	@staticmethod
	def tabulate_training(class_name, landmark_list):
		output = pd.DataFrame()
		for res in landmark_list:
			for lms in res.multi_hand_landmarks:
				map = {"class" : class_name}
				for id, lm in enumerate(lms.landmark):
					map.update({f"x{id}": lm.x, f"y{id}": lm.y, f"z{id}": lm.z})
				output = output.append(map, ignore_index=True)
		return output

	# Convert only landmarks (no class) to tabular form
	@staticmethod
	def tabulate_test(landmark_list):
		output = pd.DataFrame()
		for lms in landmark_list.multi_hand_landmarks:
			map = {}
			for id, lm in enumerate(lms.landmark):
				map.update({f"x{id}": lm.x, f"y{id}": lm.y, f"z{id}": lm.z})
			output = output.append(map, ignore_index=True)
		return output	

	# Capture training data for a particular class
	def capture_training_data(self):
		landmark_list = []
		for i in range(0, self.DEMO_LENGTH):
			res, frame = self.cap.read()
			if not res:
				print("Couldn't grab frame, continuing to next iteration")
			else:
				# Convert to RGB (opencv is natively BGR)
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				# Get hand landmarks 
				results = self.hands.process(frame)
				# Show landmarks superimposed on hand
				stat, image = self.drawHands(results, frame)
				if stat != 0:
					#print("Didn't detect hand")
					continue
				else:
					landmark_list.append(results)
					cv2.imshow("Landmarks", cv2.flip(image, 1))
					if cv2.waitKey(5) & 0xFF == 27:
						# Exit training using esacpe
						break
		cv2.destroyAllWindows()
		return landmark_list

	# Collected training data for a particular class (Have your hand ready!)
	def collect_label(self, label):
		lnd_list = self.capture_training_data()
		df = self.tabulate_training(label, lnd_list)
		return df

	# Add latest training data to overall df
	def add_training_data(self, df):
		self.overall_df = self.overall_df.append(df, ignore_index=True)

	# Write training data (overall df) to file
	def training_to_csv(self, fname="../data/training.csv"):
		self.overall_df.to_csv(fname, index=False)

if __name__ == '__main__':
	# Remove this pass statement when running actual code
	pass 

	# Usage example: 
	train = collect_train()
	df = train.collect_label("smol_pp") # You should test this one vaz
	train.add_training_data(df)
	train.training_to_csv("../data/nice_test.csv")

