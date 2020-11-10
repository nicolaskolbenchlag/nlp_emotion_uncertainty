# https://github.com/allenai/longformer
# https://huggingface.co/transformers/model_doc/longformer.html
import transformers
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import random
import sklearn.model_selection

# MODEL = "allenai/longformer-base-4096"
# TOKENIZER = "roberta-base"

MODEL = "bert-base-uncased"
TOKENIZER = MODEL
SEQ_LEN = 512

config = transformers.AutoConfig.from_pretrained(MODEL) 
config.attention_mode = "sliding_chunks"
tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER, max_length=SEQ_LEN, do_lower_case=True)# BERT
# tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER, do_lower_case=True)# Longformer

tokenizer.model_max_length = config.max_position_embeddings

def tokenize(data):
    input_ids, attention_mask = [], []
    for sentence in data:
        encoded = tokenizer.encode_plus(
            text = sentence,
            add_special_tokens = True,

            padding = "max_length",# NOTE: Remove for Longformer
            truncation = True,
            
            return_attention_mask = True,
            return_tensors = "tf"
        )
        input_ids.append(encoded.get("input_ids"))
        attention_mask.append(encoded.get("attention_mask"))
    return np.array([np.array(input_ids)[:,0,:], np.array(attention_mask)[:,0,:]])

def load_dataset(test_split=.05):
    directory_data = "../../EmCaR/"
    directory_data += "2_data/360p/transcriptions_new/"
    directory_label = "wild/label_segments/valence/"

    x, y = [], []
    for filename in os.listdir(directory_data):
        df_data = pd.read_csv(directory_data + filename, sep="\t")
        
        if filename.split("_")[0] + ".csv" not in os.listdir(directory_label):
            print("No labels file for data file {} found.".format(filename))
            continue

        df_label = pd.read_csv(directory_label + filename.split("_")[0] + ".csv", sep=",")
        df_label.timestamp = df_label.timestamp.apply(lambda ts: datetime.datetime.strptime("{}.{}.{}.{}".format(datetime.datetime.fromtimestamp(ts / 1000).hour - 1, datetime.datetime.fromtimestamp(ts / 1000).minute, datetime.datetime.fromtimestamp(ts / 1000).second, datetime.datetime.fromtimestamp(ts / 1000).microsecond), "%H.%M.%S.%f"))

        x_tmp, y_tmp = "", []
        
        last_label = 0
        for i in range(len(df_data)):
            text = df_data.label.values[i]
            
            if i > 0: text = " " + text
            x_tmp += text

            time_begin = datetime.datetime.strptime(df_data.begin_time.values[i], "%H.%M.%S.%f")
            time_end = datetime.datetime.strptime(df_data.end_time.values[i], "%H.%M.%S.%f")
            labels = df_label.loc[(df_label.timestamp >= time_begin) & (df_label.timestamp <= time_end)].value.values
            if len(labels) > 0:
                last_label = labels.mean()
            label = last_label
            
            tokens = tokenizer.encode_plus(text=text + " ", add_special_tokens=False, return_attention_mask=False, return_tensors="tf").get("input_ids")
            
            y_tmp += [label] * (tokens.shape[-1] - 1)# NOTE: -1 because of EOF-token
        
        x.append(x_tmp)

        y_tmp += [0] * (SEQ_LEN - len(y_tmp) - 2)# NOTE: for BERT
        y_tmp = y_tmp[:SEQ_LEN - 2]
        y_tmp = [0] + y_tmp + [0]
        y.append(y_tmp)
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_split)
    # x = tokenize(x)
    # y = np.array(y)
    return tokenize(x_train), tokenize(x_test), np.array(y_train), np.array(y_test)

def tilted_loss(q, y, f):
    e =  y - f
    return tf.keras.backend.mean(tf.keras.backend.maximum(q * e, (q - 1) * e), axis=-1)

def create_model(quantile, freeze_encoder=True):
    encoder = transformers.TFAutoModel.from_pretrained(MODEL)
    if freeze_encoder:
        for layer in encoder.layers:
            layer.trainable = False
    
    input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
    embedding = encoder(input_ids, attention_mask=attention_mask)[0]
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=.25, recurrent_dropout=.25)) (embedding)
    x = tf.keras.layers.Dense(1, activation="linear") (x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=x)
    model.compile(optimizer="Adam", loss=lambda y, f: tilted_loss(quantile, y, f))
    return model

def plot_prediction(seqs_real, seqs_pred):
    for i in range(len(seqs_real)):
        seq_real = seqs_real[i]
        x_axis = range(len(seq_real))
        plt.plot(x_axis, seq_real, label="Real")
        for q, seq_pred in seqs_pred.items():
            plt.plot(x_axis, seq_pred[i], label="Prediction [q={}]".format(q))
        plt.legend()
        plt.savefig("save_{}.png".format(i))
        plt.clf()
    
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_dataset()
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    assert x_train.shape[-1] == y_train.shape[-1], "Sequence lengths of x and y do not match"

    preds = {}
    for q in [.1, .5, .9]:
        model = create_model(quantile=q, freeze_encoder=False)
        model.fit([x_train[0], x_train[1]], y_train, epochs=30, batch_size=32)

        preds[q] = model.predict([x_test[0], x_test[1]])

    plot_prediction(y_test, preds)