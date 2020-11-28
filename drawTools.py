import matplotlib.pyplot as plt

def draw_lprocess(history, save=None):
    figure=plt.figure()
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(history.epoch)
    plt.plot(history.history['loss'], 'y', label='t loss')
    plt.plot(history.history['val_loss'], 'r', label='v loss')
    plt.legend(loc='upper right')
    if save!=None:
        plt.savefig(save+'_loss')        
    else:
        plt.show()
    
    figure=plt.figure()
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.xticks(history.epoch)
    plt.plot(history.history['accuracy'], 'y', label='t acc')
    plt.plot(history.history['val_accuracy'], 'r', label='v acc')
    plt.legend(loc='lower right')
    if save!=None:
        plt.savefig(save+'_acc')        
    else:
        plt.show()
