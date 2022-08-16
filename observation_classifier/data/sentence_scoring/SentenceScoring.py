from typing import Optional
import numpy as np
from loguru import logger
from datasets import Dataset
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from observation_classifier.data.sentence_scoring.DataTransformer import DatasetDict
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import pipeline, logging
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import os
os.environ["WANDB_DISABLED"] = "true" # for not signing in to anything weird


class SentenceScoring():
    def __init__(self, checkpoint:str):
        """
        Constructs all necessary attributes for the sentence scoring object.

        Input
        ---------
            checkpoint: str
                Model for tokenization and fine-tuning
        
        Returns
        ---------
            model: Pretrained model from Huggingface
        """
        logging.set_verbosity_info()
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    def compute_metrics(self, prediction):    
        logits, labels = prediction
        pred = np.argmax(logits, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def _freeze_layers(self, last_no_layers:int):
        """
        Freezing all but the last x layers

        Input
        ---------
            last_no_layers: int
                The number of layers that should not be frozen.
        """
        all_layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True: # layers that will be fine-tuned
                all_layers.append(name)

        unfreeze_layers = all_layers[-last_no_layers:]

        # Freeze all layers if not in unfreeze_layers list
        for name, param in self.model.named_parameters():
            if name not in unfreeze_layers:
                param.requires_grad = False

        print(">> Layers that are unfrozen:")
        for name, param in self.model.named_parameters():
            if param.requires_grad == True: # layers that will be fine-tuned
                print(name)

    def fine_tuning(self, train_dataset, val_dataset, train_args=None, freeze_layers=True, last_no_layers=None, checkpoint = None, opt = None, schedule = None):
        """
        Fine-tuning pretrained (BERT) model with the Trainer API

        Input:
            train_dataset and val_dataset: datasets.arrow_dataset.Dataset
                Tokenized dataset with "features" and "num_rows"
            train_args: TrainingArguments()
                Arguments used for training, e.g. learning rate, batch size, no. of epochs
            freeze_layers: bool
                freeze layers during training
            last_no_layers: int
                number for layers to train
            checkpoint: string
                Checkponint folder to continue training
            opt: transformers.optimization.
                Optimizer to use with defined learning rate scheduel
            shcedule: torch.optim.lr_scheduler.
                Learning rate schedule to use, None if standard implementation is used

        Output:
            trainer: transformers.trainer.Trainer
                Fine-tuned version of a pretrained model
        """
        if train_args == None:
            training_args = TrainingArguments("test_trainer")
        else:
            training_args = train_args

        if freeze_layers:
            self._freeze_layers(last_no_layers)

        if opt != None and schedule != None:
            trainer = Trainer(
                model = self.model, # pretrained model
                args = training_args, 
                train_dataset = train_dataset,
                eval_dataset = val_dataset,
                compute_metrics = self.compute_metrics,
                optimizers = [opt, schedule],
            )
        else:
                trainer = Trainer(
                model = self.model, # pretrained model
                args = training_args, 
                train_dataset = train_dataset,
                eval_dataset = val_dataset,
                compute_metrics = self.compute_metrics,
            )
        if checkpoint:
            trainer.train(checkpoint)
        else:
            trainer.train()

        return trainer
    
    def evaluate_model(self, trainer, save=True, dir=None):
        if save:
            eval = trainer.evaluate()
            with open(f'{dir}/evaluation.pkl', 'wb') as f:
                pickle.dump(eval, f)
        else:
            return trainer.evaluate()
    
    def predict(self, trainer, dataset, save=True, dir=None):
        if save:
            pred = trainer.predict(dataset)
            with open(f'{dir}/prediction.pkl', 'wb') as f:
                pickle.dump(pred.metrics, f)
        else:
            return trainer.predict(dataset)

    def load_model(self, dir:str, num_labels=2, trainer=True):
        model = AutoModelForSequenceClassification.from_pretrained(dir, num_labels=num_labels)
        if trainer:
            model = Trainer(model=model)
            return model
        else:
            return model


    def get_predictions(self, model, dataset):
        output = self.predict(model,dataset,save=False)
        probs = [softmax(i).tolist() for i in output.predictions]
        return np.array(probs)[:,1]

    def save_model(self, trainer, model_name: str, dir = ""):
        return trainer.save_model(f'{dir}/{model_name}')
 

if __name__ == "__main__":
    checkpoint = "Maltehb/danish-bert-botxo"
    data_dir = ''

    train_args = TrainingArguments(
        output_dir='./results',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        do_train=True,
        do_eval=True
    )

    #------------------------
    # Tokenized data
    #------------------------
    dat = DatasetDict()
    train_val = dat.load(dir=data_dir, name='tokenized_data_train_val')
    test = dat.load(dir=data_dir, name='tokenized_test')

    #------------------------
    # Sentence Scoring
    #------------------------
    ss = SentenceScoring(checkpoint=checkpoint)

    # Fine-tune and evaluate
    logger.info("Fine-tuning pretrained model")
    trainer = ss.fine_tuning(train_val, train_args=None, freeze_layers=True)
    logger.info(ss.evaluate_model(trainer))

    # Predict on unseen test set
    output = ss.predict(trainer, test)
    logger.info(f"Prediction metrics: {output.metrics}")

    # Save model
    if output.metrics["test_accuracy"] > 0.65:
        logger.info("Saving model since its accuracy is above 65%")
        ss.save_model(trainer, model_name='fine_tuned_model')