def get_confusion_matrix(model, test_dataset, y_true, true_categories, y_pred, predicted_categories):
    # Confusion Matrix
    def get_cm(model, test_dataset, y_true):

        y_prediction = model.predict(test_dataset)
        y_prediction = np.argmax(y_prediction, axis=1)
        y_test = np.argmax(y_true, axis=1)
        # Create confusion matrix and normalizes it over predicted (columns)
        result = confusion_matrix(y_test, y_prediction, normalize="pred")
        disp = ConfusionMatrixDisplay(confusion_matrix=result, display_labels=labels)
        disp.plot()
        plt.xticks(rotation=35)
        plt.savefig("confusion_matrix.png")
        plt.close()

    cm_sklearn = get_cm(model, test_dataset, y_true)
    return cm_sklearn