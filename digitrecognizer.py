# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import io
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
%autoindent

# Definição das funções necessárias para a modelagem
def registro_modelo(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def realiza_registro():
    registro_geral = pd.read_csv('registros/registros_resultados.csv')
    registro_geral = pd.concat([registro_geral, registro_atual], ignore_index=True)
    registro_geral.to_csv('registros/registros_resultados.csv', index=False) 
    with open("./registros/registros_modelagem.txt", "a") as registros:
        registros.write("########## INÍCIO DE REGISTRO - MODELO TensorFlow " + nome_modelo + " ##########\n")
        registros.write("\nInformações geradas em " + datetime.now().strftime("%d-%m-%Y") + " às " + datetime.now().strftime("%H:%M") + ".\n")
        registros.write('Parâmetros do modelo:\n')
        registros.write(registro_modelo(modelo))
        registros.write("   --> Score local do modelo (accuracy): " + str(score))
        registros.write("\n\n########## FINAL DE REGISTRO - MODELO TensorFlow " + nome_modelo + " ##########\n\n\n\n")
    print('Novo registro realizado com sucesso!')


# Definição do diretório de trabalho
os.chdir('/home/thiago/Documentos/MBA USP/Kaggle/DigitRecognizer/')

# Importação dos datasets
teste = pd.read_csv('test.csv')
treino = pd.read_csv('train.csv')

target = 'label'

# Separação x e y com normalização do x
dados_treino = treino.drop(target, axis=1)
dados_teste = treino[target]
dados_treino = dados_treino/255

# Separação dos dados de treino entre treino e validação
tamanho_treino = 0.80
treino_x, teste_x, treino_y, teste_y = train_test_split(dados_treino, dados_teste,
                                                        train_size=tamanho_treino,
                                                        random_state=1)

# Importação dos datasets direto do Keras (não precisa processar as importações
# e transformações acima
(treino_x, treino_y), (teste_x, teste_y) = tf.keras.datasets.mnist.load_data()
teste = pd.read_csv('test.csv')
teste = np.array(teste)
teste = teste.reshape(teste.shape[0], 28, 28, 1)

# Elaboração do modelo de referência
treino_x = np.array(treino_x)
treino_y = np.array(treino_y)
teste_x = np.array(teste_x)
treino_x = treino_x.reshape(treino_x.shape[0], 28, 28, 1)
teste_x = teste_x.reshape(teste_x.shape[0], 28, 28, 1)

nome_modelo = datetime.now().strftime("%Y%m%d-%H%M")
modelo = tf.keras.models.Sequential()
modelo.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), name='camada_conv_1'))
modelo.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), name='camada_conv_2'))
modelo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='camada_MaxPool_1'))
modelo.add(tf.keras.layers.Conv2D(64, kernel_size=(2, 2), name='camada_conv_3'))
modelo.add(tf.keras.layers.Conv2D(64, kernel_size=(2, 2), name='camada_conv_4'))
modelo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='camada_MaxPool_2'))
modelo.add(tf.keras.layers.Flatten())
modelo.add(tf.keras.layers.Dense(128, activation='relu', name='camada_Dense_1'))
modelo.add(tf.keras.layers.Dropout(0.2, seed=1))
modelo.add(tf.keras.layers.Dense(10, activation='softmax', name='camada_Dense_final'))
modelo.compile(optimizer='rmsprop',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Fit do modelo
meu_modelo = modelo.fit(treino_x, treino_y, epochs=50,
                        validation_data=(teste_x, teste_y), batch_size=256)

# Gráfico dos loss em função dos epochs
y_pred = modelo.predict(teste_x)
resultado = np.argmax(y_pred, axis=1)
score = round(accuracy_score(teste_y, resultado), 4)
plt.plot(meu_modelo.history['loss'])
plt.plot(meu_modelo.history['val_loss'])
plt.title(f'Loss do modelo {nome_modelo}\nem função das iterações')
plt.ylabel('Loss')
plt.xlabel(f'Iterações (epochs)\nScore (accuracy): {score}')
plt.legend(['dados de treino', 'dados de teste'])
plt.show()

# Salva o modelo realizado, realiza as previsões e apresenta o score obtido
modelo.save('modelos/'+nome_modelo+'.keras')
y_pred = modelo.predict(teste_x)
resultado = np.argmax(y_pred, axis=1)
score = round(accuracy_score(teste_y, resultado), 4)
print(f'Avaliação do modelo por Accuracy: {score}')

# Registra as informações do modelo e score em arquivos locais
registro_geral = pd.read_csv('registros/registros_resultados.csv')
registro_atual = pd.DataFrame([[pd.to_datetime(nome_modelo), score]])
registro_atual.columns = ('Dia e Hora', 'Score (accuracy)')
if registro_geral.iloc[registro_geral.shape[0]-1]['Dia e Hora'] == str(pd.to_datetime(nome_modelo)): 
    print('Trabalhando com mesmo modelo')
else:
    realiza_registro()

# Estimação dos resultados do desafio e geração do arquivo para submissão
y_final = modelo.predict(teste)
y_final = np.argmax(y_final, axis=1)
submissao = pd.read_csv('sample_submission.csv')
submissao.Label = y_final
submissao.to_csv('submissoes/submissao_TF_'+nome_modelo+'.csv', index=False)
