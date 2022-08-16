# SENTENCE SCORING

import pickle
from transformers import AutoTokenizer
from loguru import logger
import json
import os


class DatasetDict():
    """
    Load or save a dataset type
    """
    def __init__(self) -> None:
        pass
    
    def save(self, dictionary, dir, name):
        with open(f'{dir}/{name}.pkl', 'wb') as f:
            pickle.dump(dictionary, f)

    def load(self, dir, name):        
        with open(f'{dir}/{name}.pkl', 'rb') as f:
            return pickle.load(f)


class DataTransformer():
    def __init__(self, checkpoint:str):
        """
        Constructs all necessary attributes for the sentence scoring object.

        Input:
        ---------
            checkpoint: str
                Model for tokenization
        
        Returns:
        ---------
            tokenizer: Tokenizer class of the library from the pretrained model vocabulary
        """
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def _tokenization(self, dataset):
        return self.tokenizer(dataset["text"], padding="max_length", truncation=True)
    
    def tokenize(self, dataset):
        """
        Tokenize dataset

        Input:
        ---------
            dataset: datasets.arrow_dataset.Dataset
                Dataset with features "text" and "label"
        
        Returns:
        ---------
            dataset: datasets.arrow_dataset.Dataset
                Tokenized version of the dataset
        """
        return dataset.map(self._tokenization, batched=True)

 
if __name__ == "__main__":
    dir = os.getcwd()
    config_name = f'sentence_scoring_experiments_config.json'
    jsonfile = open(os.path.join(dir, config_name))
    config = json.load(jsonfile)

    # Load data
    logger.info(">> Loading data")
    file = open(f"{config['data_dir']}/{config['separator']}/{config['file_name']}", "rb")
    dataset = pickle.load(file)

    train_dataset = dataset['train']
    val_dataset = dataset['val']
    test_dataset = dataset['test']
    
    # Tokenize
    logger.info(">> Tokenizing data")
    dt = DataTransformer(config['checkpoint'])
    train_dataset = dt.tokenize(train_dataset)
    val_dataset = dt.tokenize(val_dataset)
    test_dataset = dt.tokenize(test_dataset)

    # Save data
    if config["save_datadict"]:
        logger.info(">> Saving tokenized data")
        dd = DatasetDict()
        dd.save(train_dataset, dir=f'{config["data_dir"]}/{config["separator"]}/{config["tokenized_dir"]}', name=f"tokenized_train_{config['tokenize_name']}")
        dd.save(val_dataset, dir=f'{config["data_dir"]}/{config["separator"]}/{config["tokenized_dir"]}', name=f"tokenized_val_{config['tokenize_name']}")
        dd.save(test_dataset, dir=f'{config["data_dir"]}/{config["separator"]}/{config["tokenized_dir"]}', name=f"tokenized_test_{config['tokenize_name']}")