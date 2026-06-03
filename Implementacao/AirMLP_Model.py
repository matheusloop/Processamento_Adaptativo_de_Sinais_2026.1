from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import random

def sample_hyperparameters(winSize_list, hiddenLayer_range, neuron_list, function_list, learning_rate_range, batch_size_list):

    params = {
        'winSize': random.choice(winSize_list),
        # Número de camadas ocultas
        'hiddenLayers': random.randint(hiddenLayer_range[0], hiddenLayer_range[1]),
        # Número de neurônios por camada
        'neurons': random.choice(neuron_list),
        # Função de ativação
        'function': random.choice(function_list),
        # Learning rate (log-uniform)
        'learning_rate': 10 ** random.uniform(learning_rate_range[0], learning_rate_range[1]),
        # Batch size
        'batchSize': random.choice(batch_size_list),
        # Número máximo de épocas
        'epochs': 200
    }

    return params


def createMLP_andTrain(hiddenLayers, neurons, function, learning_rate, batchSize, epochs, X_train, y_train, X_val, y_val):
    
    ##################################
    # MONTANDO A REDE NEURAL ARTIFICIAL (MLP)
    ##################################
    model = Sequential()

    model.add(Input(shape=(X_train.shape[1],)))
    
    for _ in range(hiddenLayers):
        model.add(Dense(neurons, activation=function))
    
    model.add(Dense(1))
    ##################################

    ##################################
    # COMPILANDO O MODELO
    ##################################
    model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=learning_rate))

    # Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',   # val_loss       
        patience=15,                 
        restore_best_weights=True,
        verbose=0                 
    )
    ##################################

    ##################################
    # TREINANDO O MODELO
    ##################################
    history = model.fit(
        X_train, 
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batchSize,
        callbacks=[early_stop],
        verbose=0
    )
    epochs = len(history.history['loss'])
    ##################################

    ##################################
    # CALCULANDO AS MÉTRICAS DE AVALIAÇÃO
    ##################################
    y_val_pred = model.predict(X_val)

    r2 = r2_score(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = root_mean_squared_error(y_val, y_val_pred)

    return {'epochs': epochs, 'r2': r2, 'mae': mae, 'rmse': rmse}