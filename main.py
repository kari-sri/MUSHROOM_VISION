from train import train_model, evaluate_model, train_tabular_model

if __name__ == '__main__':
    train_model(num_epochs=5)
    evaluate_model()
    train_tabular_model()
