import sys
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load
import numpy as np

accuracy = load("accuracy")
NUM_SAMPLES_TRAIN = 67345
NUM_SAMPLES_VAL = 872
NUM_SAMPLES_TEST = 1821


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def train_model(model_name, seed_limit, num_samples_train, num_samples_validation):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True)

    accuracies = []
    train_time = []
    best_acc = 0
    best_model = None
    for seed in range(seed_limit):
        train, eval = load_dataset("sst2", split=[f"train", f"validation"])

        if num_samples_train < NUM_SAMPLES_TRAIN:
            train = train.train_test_split(train_size=num_samples_train, seed=seed)["train"]

        if num_samples_validation < NUM_SAMPLES_VAL:
            eval = eval.train_test_split(train_size=num_samples_validation, seed=seed)["train"]

        train = train.map(preprocess_function, batched=True)
        eval = eval.map(preprocess_function, batched=True)

        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        trainer = Trainer(model, TrainingArguments("check",
                                                   seed=seed, data_seed=seed,
                                                   save_strategy="no",
                                                   per_device_eval_batch_size=1),
                          train_dataset=train,
                          eval_dataset=eval,
                          compute_metrics=compute_metrics,
                          tokenizer=tokenizer)

        train_time.append(trainer.train()[2]["train_runtime"])

        acc = trainer.evaluate()

        model_val_acc = acc["eval_accuracy"]
        if model_val_acc > best_acc:
            best_model = trainer
            best_acc = model_val_acc

        accuracies.append(model_val_acc)

    return np.mean(accuracies), np.std(accuracies), np.sum(train_time), best_model


def return_test_set_ready_for_pred(name_of_model, num_samples, test):
    tokenizer = AutoTokenizer.from_pretrained(name_of_model)

    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True)

    test = test.map(preprocess_function)
    if num_samples < NUM_SAMPLES_TEST:
        test = test.train_test_split(train_size=num_samples, seed=420)["train"]

    return test.remove_columns("label")


def main_func(seed_limit, num_samples_train, num_samples_validation, num_samples_test):
    res_bert = train_model("bert-base-uncased", seed_limit, num_samples_train, num_samples_validation)

    res_roberta = train_model("roberta-base", seed_limit, num_samples_train, num_samples_validation)

    res_electra = train_model("google/electra-base-generator", seed_limit, num_samples_train, num_samples_validation)

    total_train_time = res_bert[2] + res_roberta[2] + res_electra[2]

    best = np.argmax([res_bert[0], res_roberta[0], res_electra[0]])

    test = load_dataset("sst2", split=f"test")

    if best == 2:
        print("best model was google/electra-base-generator")
        preds = res_electra[3].predict(
            return_test_set_ready_for_pred("google/electra-base-generator", num_samples_test, test))

    elif best == 1:
        print("best model was roberta-base")
        preds = res_roberta[3].predict(return_test_set_ready_for_pred("roberta-base", num_samples_test, test))

    else:
        print("best model was bert-base-uncased")
        preds = res_bert[3].predict(return_test_set_ready_for_pred("bert-base-uncased", num_samples_test, test))

    results = np.argmax(preds.predictions, axis=1)
    with open("/content/drive/MyDrive/ANLP_ex1/predictions.txt", "w") as f:
        to_write = f""
        for i, pred in enumerate(results):
            to_write += f"{test['sentence'][i]}###{pred}\n"

        f.write(to_write)

    with open("/content/drive/MyDrive/ANLP_ex1/res.txt", "w") as res:
        res.write(f"bert-base-uncased,{res_bert[0]} +- {res_bert[1]}\n"
                  f"roberta-base,{res_roberta[0]} +- {res_roberta[1]}\n"
                  f"google/electra-base-generator,{res_electra[0]} +- {res_electra[1]}\n"
                  f"----\n"
                  f"train time,{total_train_time}\n"
                  f"predict time, {preds.metrics['test_runtime']}"
                  )

    return results, total_train_time


def check_parameter(param):
    if param == -1:
        return 10000000000000
    elif param < 0 and (param != -1):
        raise ValueError("All sample number arguments must be either -1 or a positive int")
    else:
        return param


if __name__ == '__main__':
    try:
        seed_range = int(sys.argv[2])
        train_samples = int(sys.argv[3])
        eval_samples = int(sys.argv[4])
        test_samlpes = int(sys.argv[5])
    except IndexError:
        raise Exception(f"Expected 4 arguments but received {len(sys.argv) - 1}")
    except ValueError:
        raise Exception(f"All arguments must be ints")

    if seed_range < 1:
        raise ValueError("Seed range parameter must be a positive int")

    train_samples = check_parameter(train_samples)
    eval_samples = check_parameter(eval_samples)
    test_samlpes = check_parameter(test_samlpes)

    main_func(seed_range, train_samples, eval_samples, test_samlpes)
