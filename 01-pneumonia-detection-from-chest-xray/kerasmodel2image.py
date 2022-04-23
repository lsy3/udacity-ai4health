import keras 
import tensorflow as tf

def load_model(model_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = keras.models.model_from_json(loaded_model_json)
    
    return model

model_path = 'my_model.json'
my_model = load_model(model_path) #loads model

print(my_model.summary())


tf.keras.utils.plot_model(my_model, to_file="my_model.png", show_shapes=True)