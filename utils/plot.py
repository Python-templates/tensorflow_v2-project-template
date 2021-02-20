import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def display_training_accuracy(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_labels):
    from sklearn.metrics import classification_report, confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15,15))
    ax = sns.heatmap(cm, cmap=plt.cm.plasma, annot=True, square=True,
                        xticklabels=class_labels, yticklabels=class_labels)
    ax.set_ylabel('Actual', fontsize=40)
    ax.set_xlabel('Predicted', fontsize=40)

    cf_report = classification_report(y_pred,y_true)
    print(cf_report)

def plot_confusion_matrix_from_dataset(dataset, model, class_labels):
    y_true = []
    y_pred = []
    iterator = dataset.as_numpy_iterator()
    # iterator.next()
    for image_batch, label_batch in iterator:
        predictions = model.predict_on_batch(image_batch)

        probabilities = tf.nn.softmax(predictions)
        predicted_indices = tf.argmax(probabilities, 1)
        target_labels = list(range(len(predictions[0])))
        predicted_class = tf.gather(target_labels, predicted_indices).numpy()
        label_class = np.where(label_batch == 1)[1]

        y_true.append(label_class)
        y_pred.append(predicted_class)

    y_true = np.hstack(np.array(y_true))
    y_pred = np.hstack(np.array(y_pred))

    plot_confusion_matrix(y_true, y_pred, class_labels)
