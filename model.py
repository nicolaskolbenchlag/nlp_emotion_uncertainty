import transformers
import torch
import torch.nn
import torch.utils.data
import os
import pandas as pd
import datetime
import time
import random
import numpy as np

MODEL = "bert-base-uncased"
SEQ_LEN = 512

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, max_length=SEQ_LEN, pad_to_max_length=True, do_lower_case=True)

def tokenize(data):
    input_ids, attention_mask = [], []
    for sentence in data:
        encoded = tokenizer.encode_plus(
            text = sentence,
            add_special_tokens = True,
            max_length = SEQ_LEN,
            truncation=True,
            pad_to_max_length = True,
            return_attention_mask = True
        )
        input_ids.append(encoded.get("input_ids"))
        attention_mask.append(encoded.get("attention_mask"))
    input_ids, attention_mask = torch.tensor(input_ids), torch.tensor(attention_mask)
    return input_ids, attention_mask

class RegressionModel(torch.nn.Module):
    def __init__(self, freeze_encoder=True):
        super(RegressionModel, self).__init__()

        self.encoder = transformers.AutoModel.from_pretrained(MODEL)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.prediction_head = torch.nn.Sequential(
            torch.nn.Linear(768, 50),
            torch.nn.ReLU(),
            torch.nn.Dropout(.25),
            torch.nn.Linear(50, 1)
        )

        self.prediction_head_2 = torch.nn.Sequential(
            torch.nn.GRU(input_size=768, hidden_size=50, num_layers=1, batch_first=False, bidirectional=True),# return sequences
            torch.nn.ReLU(),
            torch.nn.Dropout(.25),
            torch.nn.Linear(50, 1)# predict label for each timestep
        )

    def forward(self, input_ids, attention_mask):
        return self.forward_not_recurrent(input_ids, attention_mask)

    def forward_not_recurrent(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.prediction_head(last_hidden_state_cls)
        return logits
    
    def forward_recurrent(self, input_ids, attention_mask):
        outputs = []
        for i_text_step in None:
            outputs.append(self.encoder(input_ids=None, attention_mask=None))

# if torch.cuda.is_available():
if False:      
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

def initialize_model(epochs):
    model = RegressionModel()
    model.to(device)
    optimizer = transformers.AdamW(model.parameters())
    total_steps = len(train_dataloader) * epochs
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    return model, optimizer, scheduler

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

if __name__ == "__main__":
    directory_data = "../../EmCaR/"
    directory_data += "2_data/360p/transcriptions_new/"

    directory_label = "wild/label_segments/valence/"
    
    x, y = [], []

    for filename in os.listdir(directory_data)[:2]:
        df_data = pd.read_csv(directory_data + filename, sep="\t")
        
        if filename.split("_")[0] + ".csv" not in os.listdir(directory_label):
            print("No labels file for data file {} found.".format(filename))
            continue

        df_label = pd.read_csv(directory_label + filename.split("_")[0] + ".csv", sep=",")
        df_label.timestamp = df_label.timestamp.apply(lambda ts: datetime.datetime.strptime("{}.{}.{}.{}".format(datetime.datetime.fromtimestamp(ts / 1000).hour - 1, datetime.datetime.fromtimestamp(ts / 1000).minute, datetime.datetime.fromtimestamp(ts / 1000).second, datetime.datetime.fromtimestamp(ts / 1000).microsecond), "%H.%M.%S.%f"))

        # x, y = [], []

        concat_texts = 2
        for i in range(0, len(df_data), concat_texts):
            df_data_sub = df_data[i : i + concat_texts]
            
            sub_texts = df_data_sub.label.values
            text = " ".join(sub_texts)

            time_begin = datetime.datetime.strptime(df_data_sub.begin_time.values[0], "%H.%M.%S.%f")
            time_end = datetime.datetime.strptime(df_data_sub.end_time.values[-1], "%H.%M.%S.%f")
            
            labels = df_label.loc[(df_label.timestamp >= time_begin) & (df_label.timestamp <= time_end)].value.values

            if not len(labels) > 0:
                continue

            label = labels.mean()

            x.append(text)
            y.append(float(label))

    x = tokenize(x)
    max_len = max([len(text) for text in x])
    print("Max length:", max_len)

    y = torch.tensor(y)

    print("x.shape:", x.size())
    print("y.shape:", y.size())

    train_data = torch.utils.data.TensorDataset(x[0], x[1], y)
    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=32)

    epochs = 1

    loss_fn = torch.nn.MSELoss()

    set_seed(42)
    model, optimizer, scheduler = initialize_model(epochs=epochs)
    for epoch in range(epochs):
        print("Epoch:", epoch + 1)

        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            
            # Zero out any previously calculated gradients
            model.zero_grad()

            logits = model(b_input_ids, b_attn_mask)

            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(f"{epoch + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)