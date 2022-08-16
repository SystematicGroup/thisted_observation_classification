# Sentence Scoring Training Experiment
    # Load or tokenize data
    # A) Load pretrained model - fine-tune, evaluate and predict
    # B) Load own fine-tuned model - evaluate and predict
    # C) Load own fine-tuned model - predict

from observation_classifier.data.sentence_scoring.SentenceScoring import SentenceScoring
from observation_classifier.data.sentence_scoring.DataTransformer import DataTransformer, DatasetDict
from transformers import AdamW, TrainingArguments, get_cosine_with_hard_restarts_schedule_with_warmup
from loguru import logger
import json
import pickle
import os
from datetime import datetime
os.environ["WANDB_DISABLED"] = "true" # for not signing in to anything weird

if __name__=="__main__":
    dir = os.getcwd()
    dir = os.path.join(dir , 'observation_classifier/models/sentence_scoring')
    config_name = f'sentence_scoring_experiments_config.json'
    jsonfile = open(os.path.join(dir, config_name))
    config = json.load(jsonfile)

    if config["tokenize_data"]:
        # Load data
        logger.info(">> Loading tokenized data")
        file = open(f"{config['data_dir']}/{config['separator']}/{config['file_name']}", "rb")
        dataset = pickle.load(file)

        train_dataset = dataset['train']
        val_dataset = dataset['val']
        test_dataset = dataset['test']
        
        # Tokenize
        dt = DataTransformer(config["checkpoint"])
        train_dataset = dt.tokenize(train_dataset)
        val_dataset = dt.tokenize(val_dataset)
        test_dataset = dt.tokenize(test_dataset)

    else:
        # Load tokenized data
        dd = DatasetDict()
        train_dataset = dd.load(dir=f'{config["data_dir"]}/{config["separator"]}/{config["tokenized_dir"]}', name=f'tokenized_train_{config["tokenize_name"]}')
        val_dataset = dd.load(dir=f'{config["data_dir"]}/{config["separator"]}/{config["tokenized_dir"]}', name=f'tokenized_val_{config["tokenize_name"]}')
        test_dataset = dd.load(dir=f'{config["data_dir"]}/{config["separator"]}/{config["tokenized_dir"]}', name=f'tokenized_test_{config["tokenize_name"]}')



    #----------------------------------------------------
    # A) Sentence scoring - train, evaluate and predict
    #----------------------------------------------------
    logger.info(">> Sentence scoring")

    now = datetime.now()
    now_date = now.strftime("%d.%m.%Y_%H:%M:%S")
    config['logging_dir'] = os.path.join(config['output_dir'], config['logging_dir'] , now_date)
    train_args = TrainingArguments(
        output_dir=config['output_dir'],
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        num_train_epochs=config['num_train_epochs'],
        weight_decay=config['weight_decay'],
        do_train=True,
        do_eval=True,
        save_steps= 5000,
        eval_steps= config["evaluation_steps"],
        evaluation_strategy=config["evaluation_strategy"],
        logging_first_step=True,
        logging_dir = config['logging_dir'],
        logging_steps= config["logging_steps"]
    )
    

    # Fine-tune and evaluate
    logger.info("Fine-tuning pretrained model")
    
    ss = SentenceScoring(checkpoint=config["checkpoint"])
    if config['lr_schedule'] == 'cosine_with_restart':
        opt = AdamW(ss.model.parameters(), lr=config['learning_rate'])
        schedule = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer = opt, num_warmup_steps = config['num_warmup_steps'], num_training_steps = (round(len(train_dataset)/config['per_device_train_batch_size']*config['num_train_epochs'])), num_cycles = config['num_cycles'])

        trainer = ss.fine_tuning(
            train_dataset,
            val_dataset,
            train_args=train_args,
            freeze_layers=config['freeze_layers'],
            last_no_layers=config['last_no_layers'],
            opt = opt,
            schedule = schedule
        )
    elif config['lr_schedule'] == None:
            trainer = ss.fine_tuning(
            train_dataset,
            val_dataset,
            train_args=train_args,
            freeze_layers=config['freeze_layers'],
            last_no_layers=config['last_no_layers']
        )

    logger.info(">> Evaluating model:")
    logger.info(ss.evaluate_model(trainer, save=config['save_metrics'], dir=f'{config["data_dir"]}/{config["separator"]}'))

    # Predict on unseen test set
    output = ss.predict(trainer, test_dataset, save=config['save_metrics'], dir=f'{config["data_dir"]}/{config["separator"]}')
    logger.info(f"Prediction metrics: {output.metrics}")

    today = datetime.now()
    d1 = today.strftime("%d_%m_%Y_%H:%M")

    # Save model
    if output.metrics["test_accuracy"] >= 0.50:
        logger.info("Saving model since its accuracy is above 50%")
        ss.save_model(trainer, model_name=f"Epoch_{config['num_train_epochs']}_lr_{config['learning_rate']}_wd_{config['weight_decay']}_freezing-layer{config['last_no_layers']}_{d1}", dir=config['output_dir'])
    else:
        logger.info("Not saving model, since accuracy is less than 50%")
