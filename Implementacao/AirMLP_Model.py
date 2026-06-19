# Importação das bibliotecas necessárias
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import random


def sample_hyperparameters(winSize_list, hiddenLayer_range, neuron_list,
                           function_list, learning_rate_range, batch_size_list):
    """
    Sorteia aleatoriamente uma combinação de hiperparâmetros para a rede MLP.
    """

    params = {
        # Tamanho da janela utilizada na entrada dos dados
        'winSize': random.choice(winSize_list),

        # Número de camadas ocultas da rede
        'hiddenLayers': random.randint(hiddenLayer_range[0], hiddenLayer_range[1]),

        # Número de neurônios em cada camada oculta
        'neurons': random.choice(neuron_list),

        # Função de ativação das camadas ocultas
        'function': random.choice(function_list),

        # Taxa de aprendizado (escala logarítmica)
        'learning_rate': 10 ** random.uniform(learning_rate_range[0], learning_rate_range[1]),

        # Número de amostras processadas a cada atualização dos pesos
        'batchSize': random.choice(batch_size_list),

        # Número máximo de épocas de treinamento
        'epochs': 200
    }

    return params


def createMLP_andTrain(hiddenLayers, neurons, function,
                       learning_rate, batchSize, epochs,
                       X_train, y_train, X_val, y_val):
    """
    Cria, treina e avalia uma rede neural MLP.
    Retorna o número de épocas efetivamente utilizadas e
    as métricas de desempenho no conjunto de validação.
    """

    ###################################################
    # CONSTRUÇÃO DA REDE NEURAL
    ###################################################

    # Cria um modelo sequencial
    model = Sequential()

    # Define a camada de entrada com o número de atributos de X_train
    model.add(Input(shape=(X_train.shape[1],)))

    # Adiciona as camadas ocultas
    for _ in range(hiddenLayers):
        model.add(Dense(neurons, activation=function))

    # Camada de saída com um neurônio (problema de regressão)
    model.add(Dense(1))

    ###################################################
    # COMPILAÇÃO DO MODELO
    ###################################################

    # Utiliza erro absoluto médio (MAE) como função de perda
    # e o otimizador Adam com a taxa de aprendizado especificada
    model.compile(
        loss='mean_absolute_error',
        optimizer=Adam(learning_rate=learning_rate)
    )

    ###################################################
    # CONFIGURAÇÃO DO EARLY STOPPING
    ###################################################

    # Interrompe o treinamento caso a perda de validação
    # não apresente melhora por 15 épocas consecutivas
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )

    ###################################################
    # TREINAMENTO DO MODELO
    ###################################################

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batchSize,
        callbacks=[early_stop],
        verbose=0
    )

    # Número real de épocas executadas
    epochs = len(history.history['loss'])

    ###################################################
    # AVALIAÇÃO DO MODELO
    ###################################################

    # Realiza previsões no conjunto de validação
    y_val_pred = model.predict(X_val)

    # Coeficiente de determinação (R²)
    r2 = r2_score(y_val, y_val_pred)

    # Erro absoluto médio (MAE)
    mae = mean_absolute_error(y_val, y_val_pred)

    # Raiz do erro quadrático médio (RMSE)
    rmse = root_mean_squared_error(y_val, y_val_pred)

    # Retorna as métricas obtidas
    return {
        'epochs': epochs,
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }