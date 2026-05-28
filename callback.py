# Alternate way to implement call back 

import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs = None):
        """
        Halts the training when the loss falls below 0.4

        Args: 
        epoch (integer) - index of epoch (required but remains unsused in the function defination below)

        logs (dict) - metric results form the training epoch
        """
        if logs["loss"] < 0.4:
            print("\n Loss is lower than 0.4 so cancelling the training!")
            self.model.stop_training = True



