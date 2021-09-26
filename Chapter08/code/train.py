import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer
 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
 
def get_data_loader(batch_size, training_dir, filename, MAX_LEN):
    logger.info("Get data loader")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
 
    dataset = pd.read_csv(os.path.join(training_dir, filename))
    articles = dataset.article.values
    sentiments = dataset.sentiment.values
 
    input_ids = []
    for sent in articles:
        encoded_articles = tokenizer.encode(sent, add_special_tokens=True)
        input_ids.append(encoded_articles)
 
    # pad shorter sentences
    input_ids_padded = []
    for i in input_ids:
        while len(i) < MAX_LEN:
            i.append(0)
        input_ids_padded.append(i)
    input_ids = input_ids_padded
 
    # mask; 0: added, 1: otherwise
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
 
    # convert to PyTorch data types.
    train_inputs = torch.tensor(input_ids)
    train_labels = torch.tensor(sentiments)
    train_masks = torch.tensor(attention_masks)
 
    tensor_data = TensorDataset(train_inputs, train_masks, train_labels)
    
    tensor_dataloader = DataLoader(tensor_data, batch_size=batch_size)
 
    return tensor_dataloader
 
def train(args):
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")
 
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
 
    train_loader = get_data_loader(args.batch_size, args.data_dir, args.train_file, args.MAX_LEN)
    test_loader = get_data_loader(args.test_batch_size, args.test, args.test_file, args.MAX_LEN)
 
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  
        num_labels=args.num_labels, 
        output_attentions=False,  
        output_hidden_states=False,  )
 
    model = model.to(device) # load the model to the right device
    
    # configure optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr  # learning rate 
    )
 
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_loader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
 
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
 
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
         
            optimizer.step()
 
            if step % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        step * len(batch[0]),
                        len(train_loader.sampler),
                        100.0 * step / len(train_loader),
                        loss.item(),
                    )
                )
 
        logger.info("Average training loss: %f\n", total_loss / len(train_loader))
 
        test(model, test_loader, device)
 
    logger.info("Saving tuned model.")
    model_2_save = model.module if hasattr(model, "module") else model
    model_2_save.save_pretrained(save_directory=args.model_dir)
    
    return model
 
def test(model, test_loader, device):
    
    def get_correct_count(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat), len(labels_flat)
    
    model.eval()
    _, eval_accuracy = 0, 0
    total_correct = 0
    total_count = 0
 
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
 
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            preds = outputs[0]
            preds = preds.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
                        
            num_correct, num_count = get_correct_count(preds, label_ids)
            total_correct += num_correct
            total_count += num_count
            
    logger.info("Test set: Accuracy: %f\n", total_correct/total_count)
 
    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
 
    # Data and model checkpoints directories
    parser.add_argument(
        "--num_labels", type=int, default=3, metavar="N", help="input batch size for training (default: 3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=100, metavar="N", help="input batch size for testing (default: 100)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 2)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--train_file", type=str, default=None, help="train file name")
    parser.add_argument("--test_file", type=str, default=None, help="test file name")
    parser.add_argument("--MAX_LEN", type=int, default=300, help="max length of the input str")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    
    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
 

    train(parser.parse_args())
