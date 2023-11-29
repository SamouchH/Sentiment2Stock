#Importing librairies
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import  get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import GradScaler, autocast

#Set the device to the GPU
torch.cuda.set_device(0)

#Load and Preprocess Data
df = pd.read_csv('./data/stock_tweets_with_sentiment.csv')

#define a function to categorize sentiment scores
def categorize_sentiment(score):
    if score <= 0.4:
        return 0
    elif score <= 0.6:
        return 1
    else:
        return 2

#Apply the function to create a new column for sentiment labels
df['sentiment_label'] = df['compound_score'].apply(categorize_sentiment)

#Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

## Tokenization and Data Preparation
#Defining the dataset class fro handling tokenization and formatting
class TweetDataset(Dataset):
    """
    A custom PyTorch Dataset class to handle tokenization and formatting of the tweet data

    """
    def __init__(self, tweets, labels, tokenizer, max_length):
        self.tweets = tweets
        self.labels = labels 
        self.tokenizer = tokenizer #BERT tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        label= self.labels[item]

        #Tokenize and encode the tweet text
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        #Return the input_ids, attention_mask, and labels as tensors
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
#Function to create dataloaders
def create_data_loader(df, tokenizer, max_length, batch_size):
    ds = TweetDataset(
        tweets=df['Tweet'].to_numpy(),
        labels=df['sentiment_label'].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True
    )

#Model Training Function
def train_model(model, data_loader, optimizer, scheduler, device):
    
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    scaler= GradScaler()
    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(data_loader, leave=True)
        #Loop over the training data in batches
        for batch in loop:
            #Unpack the training batch from the dataloader and copy each tensor to the GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            #Clear any previously calculated gradients before performing a backward pass
            model.zero_grad()
            

            with autocast():
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                #Get the loss and logits
                loss = outputs.loss
                logits = outputs.logits


            # Scale the loss and call backward
            scaler.scale(loss).backward()

            # Unscale the gradients and call optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            #Update the weights and learning rate            
            scheduler.step()

            #Update the progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

        #Save a checkpoint at the end of each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # Include any other information you want
        }
        torch.save(checkpoint,f'./models_LLM/model_checkpoint_epoch_{epoch+1}.pth')

    #Save the final model
    torch.save(model.state_dict(), './models_LLM/model.pth')

#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

#Export the profile results to json
#prof.export_chrome_trace("trace.json")


#Model Evaluation
#Function to evaluate the model
def evaluate_model(model, data_loader, device):
    model= model.eval()

    predictions, labels = [], []
    loop = tqdm(data_loader, leave=True)

    with torch.no_grad():
        for batch in loop:
            #Unpack the batch and copy each tensor to the GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels'].to(device)

            #Perform a forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            #Get the predicted class
            _, preds = torch.max(outputs.logits, dim=1)

            #Move the preds and labels to the CPU
            preds = preds.detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()

            #Store the predictions and labels
            predictions.extend(preds)
            labels.extend(label_ids)

        #Calculate and return the evaluation metrics
        print('Classification Report:', classification_report(labels, predictions, target_names=['Negative', 'Neutral', 'Positive'], digits=4))


if __name__ == '__main__':
    #Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #Create the dataloaders
    BATCH_SIZE = 16
    MAX_LEN = 160
    train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

    #Load the pre-trained BERT model for sequence classification
    NUM_LABELS = 3

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', #Use the 12-layer BERT model, with an uncased vocab
        num_labels=NUM_LABELS, #The number of output labels
        output_attentions=False, #Whether the model returns attentions weights
        output_hidden_states=False #Whether the model returns all hidden-states
    )

    #Move the model to the GPU
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
    print("Using device:", device)
    model.to(device)




    EPOCHS =  10
    #Define Optimizer and Loss Function
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * EPOCHS

    #Training Scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    
    #Train the model
    train_model(model, train_data_loader, optimizer, scheduler, device)

    #Evaluate the model on the test set
    evaluate_model(model, test_data_loader,device)

