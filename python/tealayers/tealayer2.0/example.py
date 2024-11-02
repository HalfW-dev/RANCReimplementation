from tealayer2 import Tea, AdditivePooling
from tensorflow.keras.layers import Flatten, Activation, Input, Lambda, concatenate
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model

import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.optimizers.legacy import Adam


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
            
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Greyscale images are of shape (28,28,1)
inputs = Input(shape=(28,28,1))

# Flatten the inputs so that inputs map as: flatten_input[0] -> axon[0], ..., flatten_input[255] -> axon[255]
flattened_inputs = Flatten()(inputs)

# Generate each core.
# We are taking a 16x16 square of the input image and striding it by 12. this gives us 4 cores with 0 padding encumpassing the entire image.
core0 = Lambda(lambda x : x[:, :256])(flattened_inputs)
core1 = Lambda(lambda x : x[:, 176:432])(flattened_inputs)
core2 = Lambda(lambda x : x[:, 352:608])(flattened_inputs)
core3 = Lambda(lambda x : x[:, 528:])(flattened_inputs)

# Use the image distributions as corresponding inputs into our Tea Layer.
core0 = Tea(units=64, name='tea_1_1')(core0)
core1 = Tea(units=64, name='tea_1_2')(core1)
core2 = Tea(units=64, name='tea_1_3')(core2)
core3 = Tea(units=64, name='tea_1_4')(core3)

# The classification is the concatenation of these 4 core's outputs.
# We'll call the classification core our 'network'
network = concatenate([core0, core1, core2, core3])

network = Tea(units=250, name='tea_2')(network)

network = AdditivePooling(10)(network)

predictions = Activation('softmax')(network)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=0)

print("Test Loss: ", score[0])
print("Test Accuracy: ", score[1])

# Optionally, then save the generated network out for use in the simulator and/or hardware
from rancutils.teaconversion import create_cores, create_packets, Packet
from rancutils.output_bus import OutputBus
from rancutils.serialization import save as sim_save

x_test_flat = X_test.reshape((10000, 784))
partitioned_packets = []

# Use absolute/hard reset by specifying neuron_reset_type=0
cores_sim = create_cores(model, 2, neuron_reset_type=0)
num_test_samples = 100
# Partition the packets into groups as they will be fed into each of the input cores
partitioned_packets.append(x_test_flat[:num_test_samples, :256])
partitioned_packets.append(x_test_flat[:num_test_samples, 176:432])
partitioned_packets.append(x_test_flat[:num_test_samples, 352:608])
partitioned_packets.append(x_test_flat[:num_test_samples, 528:])
packets_sim = create_packets(partitioned_packets)
output_bus_sim = OutputBus((0, 2), num_outputs=250)

# This file can then be used as an input json to the RANC Simulator through the "input file" argument.
sim_save("./toy/mnist_config.json", cores_sim, packets_sim, output_bus_sim, indent=2)

predict = model.predict(X_train[:num_test_samples,:])
idx = []
for i in predict:
  idx.append(np.argmax(i))
test_predictions = to_categorical(idx)

# Additionally, output the tensorflow predictions and correct labels for later cross validation
np.savetxt("./toy/mnist_tf_preds.txt", test_predictions, delimiter=',', fmt='%i')
np.savetxt("./toy/mnist_correct_preds.txt", y_test, delimiter=',', fmt='%i')

from rancutils.emulation import output_for_testbench, output_for_streaming

output_for_streaming(cores_sim,max_xy=(4,2),output_path="./toy/mnist_1_core_mem")

output_for_testbench(packets_sim,
                         y_test[:num_test_samples,:],
                         output_path='./toy',
                         input_filename='tb_input.txt',
                         correct_filename='tb_correct.txt',
                         num_inputs_filename='tb_num_inputs.txt',
                         num_outputs_filename='tb_num_outputs.txt',
                         max_packet_xy=(512, 512),
                         num_axons=256,
                         num_ticks=16,
                         num_outputs=250)

layer_name = model.layers[-3].name
intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_model.predict(X_train[:num_test_samples,:])
#print(intermediate_output)

np.savetxt('./toy/python_output.txt', intermediate_output, fmt='%i')

# TODO: Add usage example for outputting to emulation via rancutils.emulation.write_cores, etc.