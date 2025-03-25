import numpy as np
import pickle

# Loading the model
loaded_model = pickle.load(open('model.sav', 'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

# Changing the input data to numpy array
array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
  print('The person is not diabetic')
else:
  print('The person is diabetic')