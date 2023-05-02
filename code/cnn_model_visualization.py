import visualkeras
from keras.utils.vis_utils import plot_model
from keras.models import load_model

def visualization_cnn(seed=4012, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False, epochs=100, model_num=0, good='NO'):
    if model_type == 'CNN':
        if good == 'NO':
            modelname = '../model/new_PCA_model/cnn_model' + str(cnn) + '_seed' + str(seed) + '_epochs' + str(epochs) +\
                '_days' + str(n_days) + '_stride' + str(stride) + \
                '_diff' + str(diff) + '.h5'
        elif good == 'good':
            modelname = '../model/new_PCA_model/cnn_model' + str(cnn) + '_seed' + str(seed) + '_epochs' + str(epochs) +\
                '_days' + str(n_days) + '_stride' + str(stride) + \
                '_diff' + str(diff) + '_good.h5'
    elif model_type == 'LSTM':
        surname = f'lstm_model{model_num}_seed{seed}'
        modelname = '../model/' + surname + '.h5'

    model = load_model(modelname)
    filename = '../graph/report/cnn_model' + str(cnn) + '.png'
    plot_model(model, to_file=filename, show_shapes=True)


if __name__ == '__main__':
    visualization_cnn(seed = 0, cnn=1, n_days=5, stride=1, model_type='CNN', diff=True)
    visualization_cnn(seed = 0, cnn=2, n_days=5, stride=1, model_type='CNN', diff=True)
    visualization_cnn(seed = 0, cnn=3, n_days=5, stride=1, model_type='CNN', diff=True)
    visualization_cnn(seed = 0, cnn=4, n_days=5, stride=1, model_type='CNN', diff=True)

    # import pydot
    # import graphviz
    # from keras.models import Sequential
    # from keras.layers import Conv1D, Dense, Flatten
    # from keras.optimizers import Adam
    # import visualkeras

    # def cnn_model1(n_days):
    #     model = Sequential()
    #     model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=(n_days, 5)))
    #     model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
    #     model.add(Flatten())
    #     model.add(Dense(128, activation='relu'))
    #     model.add(Dense(16, activation='relu'))
    #     model.add(Dense(1))
    #     model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['accuracy'])
    #     return model

    # model = cnn_model1(n_days=5)
    # plot_model(model,  show_shapes=True)
    # visualkeras.layered_view(model).show()

