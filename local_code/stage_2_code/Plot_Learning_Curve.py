import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_learning_curves(histories, file_path, title='MLP learning convergence'):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        ax_loss.plot(epochs, history['train_loss'], label=label)
        ax_acc.plot(epochs, history['train_acc'], label=label)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Training loss')
    ax_loss.set_title('Training loss')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Training accuracy')
    ax_acc.set_title('Training accuracy')
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    print('saved learning curve to', file_path)
