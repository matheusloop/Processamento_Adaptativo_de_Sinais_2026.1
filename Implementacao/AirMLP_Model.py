from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import random

def sample_hyperparameters():

    params = {
        # Número de camadas ocultas
        'nHiddenLayers': random.randint(4, 10),
        # Número de neurônios por camada
        'nNeurons': random.choice([64, 128, 256, 512]),
        # Função de ativação
        'function': random.choice(['relu', 'tanh']),
        # Learning rate (log-uniform)
        'learning_rate': 10 ** random.uniform(-5, -2),
        # Batch size
        'batchSize': random.choice([64, 128, 256, 512]),
        # Número máximo de épocas
        'nEpochs': random.choice([100])
    }

    return params


def createMLP_andTrain(nHiddenLayers, nNeurons, function, learning_rate, batchSize, nEpochs, X_train, y_train, X_val, y_val):
    
    ##################################
    # MONTANDO A REDE NEURAL ARTIFICIAL (MLP)
    ##################################
    model = Sequential()

    model.add(Input(shape=(X_train.shape[1],)))
    
    for _ in range(nHiddenLayers):
        model.add(Dense(nNeurons, activation=function))
    
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
        epochs=nEpochs,
        batch_size=batchSize,
        callbacks=[early_stop],
        verbose=0
    )
    nEpochs = len(history.history['loss'])
    ##################################

    ##################################
    # CALCULANDO AS MÉTRICAS DE AVALIAÇÃO
    ##################################
    y_val_pred = model.predict(X_val)

    r2 = r2_score(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = root_mean_squared_error(y_val, y_val_pred)

    return {'nEpochs': nEpochs, 'r2': r2, 'mae': mae, 'rmse': rmse}