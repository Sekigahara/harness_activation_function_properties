import os
import matplotlib.pyplot as plt

from module.utils import directory_checker

class Evaluation:
    def __init__(self) -> None:
        pass

    def plot_loss(self, training_loss_list:list, testing_loss_list:list, save_path:str=''):
        fig = plt.gcf()
        plt.plot(training_loss_list)
        plt.plot(testing_loss_list)
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        if save_path != '':
            directory_checker(save_path)
            save_path = os.path.join(save_path, 'loss.jpg')
            fig.savefig(save_path)

    def plot_accuracy(self, training_accuracy_list:list, testing_accuracy_list:list, save_path:str=''):
        fig = plt.gcf()
        plt.plot(training_accuracy_list)
        plt.plot(testing_accuracy_list)
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        if save_path != '':
            directory_checker(save_path)
            save_path = os.path.join(save_path, 'accuracy.jpg')
            fig.savefig(save_path)

    def plot_gradient(self, gradient_history, loss_history, save_path:str=''):
        fig, ax = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(8, 12))
        
        ax[0].set_title("Mean Gradient")
        for key in gradient_history[0]:
            ax[0].plot(range(len(gradient_history)), [w[key].mean() for w in gradient_history], label=key)
        ax[0].legend()

        ax[1].set_title("S.D.")
        for key in gradient_history[0]:
            ax[1].semilogy(range(len(gradient_history)), [w[key].std() for w in gradient_history], label=key)
        ax[1].legend()

        ax[2].set_title("Loss")
        ax[2].plot(range(len(loss_history)), loss_history)
        plt.show()

        if save_path != '':
            directory_checker(save_path)
            save_path = os.path.join(save_path, 'gradient_information.jpg')
            fig.savefig(save_path)