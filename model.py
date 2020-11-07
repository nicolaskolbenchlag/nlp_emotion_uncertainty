import transformers
import torch
import torch.nn
import os
import pandas as pd

MODEL = "bert-base-cased"
SEQ_LEN = 512

class RegressionModel(torch.nn.Module):
    def __init__(self, freeze_encoder=True):
        super(RegressionModel, self).__init__()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, max_length=SEQ_LEN, pad_to_max_length=True)
        self.encoder = transformers.TFAutoModel.from_pretrained(MODEL)

        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        self.prediction_head = torch.nn.Sequential(
            torch.nn.Linear(768, 50),
            torch.nn.ReLU(),
            torch.nn.Dropout(.5),
            torch.nn.Linear(50, 1)
        )

    def preprocess(self, sentence):
        return sentence
    
    def tokenize(self, data):
        input_ids, attention_mask = [], []

        for sentence in data:
            encoded = self.tokenizer.encode_plus(
                text = self.preprocess(sentence),
                add_special_tokens = True,
                max_length = SEQ_LEN,
                pad_to_max_length = True,
                return_attention_mask = True
            )

            input_ids.append(encoded.get("input_ids"))
            attention_mask.append(encoded.get("attention_mask"))
        
        input_ids, attention_mask = torch.tensor(input_ids), torch.tensor(attention_mask)
        return input_ids, attention_mask


    def forward(self, input_ids, attention_mask):
        pass

if __name__ == "__main__":
    # if torch.cuda.is_available():       
    #     device = torch.device("cuda")
    #     print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    #     print("Device name:", torch.cuda.get_device_name(0))
    # else:
    #     print("No GPU available, using the CPU instead.")
    #     device = torch.device("cpu")
        
    directory_data = "../../EmCaR/"
    directory_data += "2_data/360p/transcriptions_new/"

    directory_label = "wild/label_segments/valence/"
    
    # texts = []
    for filename in os.listdir(directory_data):
        df_data = pd.read_csv(directory_data + filename, sep="\t")

        df_label = pd.read_csv(directory_label + filename.split("_")[0] + ".csv", sep=",")
        break