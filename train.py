import tensorflow as tf


def train_model(model, data, batch_size, epochs, es_patience, callbacks=[]):
    x_train, y_train, x_valid, y_valid = data['x_train_set'], data['y_train_set'], data['x_valid_set'], data['y_valid_set']

    model.summary()

    callbacks.extend([
        tf.keras.callbacks.EarlyStopping(monitor='val_output_loss', patience=es_patience, restore_best_weights=True,
                                         verbose=1),
    ])
    
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=epochs,
                        callbacks=callbacks)
    
    return model
