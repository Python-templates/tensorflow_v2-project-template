from data_loader.example_data_loader import DatasetGenerator
from models.example_model import ExampleModel
from trainers.example_train import ExampleTrainer
from utils.utils import get_args
from utils.config import process_config


def main():

    config = process_config("configs/example_config.json")

    train_data, test_data = DatasetGenerator(config)()
    model = ExampleModel()
    trainer = ExampleTrainer(model, train_data, test_data, config)
    trainer.train()


if __name__ == '__main__':
    main()


