from data_loader.example_data_loader import DatasetGenerator
from models.example_model import ExampleModel
from trainers.example_train import ExampleTrainer
from utils.dirs import create_dirs
from utils.config import process_config


def main():

    config = process_config("configs/example_config.json")
    create_dirs([config.summary_dir, config.checkpoint_dir])

    train_data, test_data = DatasetGenerator(config)()
    model = ExampleModel()
    trainer = ExampleTrainer(model, train_data, test_data, config)
    trainer.train()


if __name__ == '__main__':
    main()


