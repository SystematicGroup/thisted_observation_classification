import pickle
import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import Dataset
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


class Model():
    def __init__(self):
        pass

    def set_model(self, model):
        self.model = model


class SklearnModel(Model):
    def __init__(self):
        super().__init__()
    
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.model = self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame):
        y_pred = self.model.predict(X_test)
        return y_pred

    def save_model(self, dir, model_name):
        pickle.dump(self.model, open(f'{dir}/{model_name}.pkl'), 'wb')
  
    def load_model(self, dir, model_name):
        self.model = pickle.load(open(f'{dir}/{model_name}', 'rb'))


class HuggingfaceModel(Model):
    def __init__(self):
        super().__init__()
    
    def pd_to_dataset(self, df_traineval, df_test):
        df_traineval = Dataset(pa.Table.from_pandas(df_traineval))
        df_test = Dataset(pa.Table.from_pandas(df_test))
        return df_traineval, df_test

    def compute_metrics(self, prediction):    
        logits, labels = prediction
        is_binary_classification = (len(np.unique(labels))==2)
        if is_binary_classification:
            avg = 'binary'
        else :
            avg = 'macro'
        pred = np.argmax(logits, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average=avg)
        precision = precision_score(y_true=labels, y_pred=pred, average=avg)
        f1 = f1_score(y_true=labels, y_pred=pred, average=avg)
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

    def train(
        self,
        args,
        train_dataset,
        val_dataset,
        train_args
             
    ):
        """Fine-tune transformers model

        Input:
            model (_type_): _description_
            train_dataset (datasets.arrow_dataset.Dataset): Tokenized dataset with "features" and "num_rows"
            val_dataset (datasets.arrow_dataset.Dataset): Tokenized dataset with "features" and "num_rows"
            train_args (_type_, optional): TrainingArguments(). Arguments used for training, e.g. learning rate, batch size, no. of epochs- Defaults to None.
            freeze_layers (bool, optional): _description_. Defaults to True.
            last_no_layers (_type_, optional): _description_. Defaults to None.
            checkpoint (_type_, optional): _description_. Defaults to None.

        Output:
            _type_: _description_
        """

        training_args = TrainingArguments(
            output_dir= args['checkpoint_dir'],
            learning_rate=train_args['lr'], 
            per_device_train_batch_size=train_args['bs'],
            per_device_eval_batch_size=train_args['bs'],
            num_train_epochs=train_args['num_epochs'],
            weight_decay=train_args['weight_decay'],
            do_train=True,
            do_eval=True,
            save_steps= 5000
        )

        if train_args['freeze_layers']:
            self._freeze_layers(train_args['last_no_layers'])

        trainer = Trainer(
            model = self.model, # pretrained model
            args = training_args, 
            train_dataset = train_dataset,
            eval_dataset = val_dataset,
            compute_metrics = self.compute_metrics,
        )
        
        mlflow.end_run()
        mlflow.start_run(nested=True)

        trainer.train()

        return trainer

    def evaluate_model(self, trainer):
        return trainer.evaluate()

    def predict(self, test_dataset, trainer, get_probs=False):
        """Predict from a testset

        Args:
            trainer (_type_): _description_
            test_dataset (_type_): _description_
            get_probs (bool, optional): Get probability scores for being positive. Defaults to True.

        Output:
            _type_: _description_
        """
        pred = trainer.predict(test_dataset)

        if get_probs:
            probs = [softmax(i).tolist() for i in pred.predictions]
            return probs

        return pred

    def save_model(self, model, dir, model_name: str):
        return model.save_model(f'{dir}/{model_name}')
    
    def load_hf_model(self, num_labels=67):
        """Loading pretrained model from Huggingface"""
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model, num_labels=num_labels)


    def load_model(self, dir:str, model_name, training_args = {}, num_labels=2, trainer=True):
        """Load fine-tuned huggingface model from directory

        Args:
            dir (str): Directory to fine-tuned model
            num_labels (int, optional): Number of labels available. Defaults to 2.
            trainer (bool, optional): _description_. Defaults to True.

        Output:
            AutoModel: model
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(f'{dir}/{model_name}', num_labels=num_labels, ignore_mismatched_sizes=True)
        if trainer:
            self.model = Trainer(model=self.model)
        return self.model
