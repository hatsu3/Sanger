import json
import os
import argparse
from pathlib import Path

from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM


""" CLOTH Dataset """


BLANK_ID = 1035  # bert_uncased for "_"
MASK_ID = 103  # for BERT
SEP_TOKEN = 102
DATASET_ROOT = 'data/cloth'


class ClozeDataset(Dataset):
    """
    the simplest data format: {article, options, answers}
    article and answers zero padded, options -1 padded
    options that contains multiple tokens are truncated
     94 articles longer than 512, articles that are much too long are not discarded here, but will be truncated by my BERT model. The ignored options are filled with A.
     5677 answers contains more than 1 BERT tokens, but only 2 of them cannot be disinguished using the initial token
     for BERT, BLANK_ID should be changed into [MASK]
    """

    def __init__(self, data_list, max_len, tokenizer):
        super().__init__()
        self.data = []
        self.meta = []
        self.max_len = max_len
        self.tokenizer = tokenizer
        # how many answers contain multiple bert tokens?
        cnt = 0
        cnt1 = 0
        # how many cannot be distinguished by the initial token?
        cnt2 = 0
        for item in tqdm(data_list):
            # article
            article = item["article"].lower()
            article = self.tokenizer.encode(article)
            length = len(article)
            article = torch.tensor(article)
            n_blanks_before = sum(article == BLANK_ID)
            if length > self.max_len:
                cnt1 += 1
                article = article[:self.max_len]
                article[-1] = SEP_TOKEN
            n_blanks = sum(article == BLANK_ID)
            article = (article * (article != BLANK_ID).long()) + (
                MASK_ID * (article == BLANK_ID).long()
            )

            # answers
            answers = [self.foo(i) - self.foo("A") for i in item["answers"]][:n_blanks]
            answers = torch.tensor(answers)

            # options
            options = [
                [self.tokenizer.encode(word)[1:-1] for word in line]
                for line in item["options"]
            ][:n_blanks]
            for i, option in enumerate(options):
                if answers.shape[0] > 0:
                    if len(option[answers[i]]) > 1:
                        cnt += 1
                        if (
                            option[answers[i]]
                            in option[0 : answers[i]] + option[answers[i] + 1 :]
                        ):
                            cnt2 += 1
                options[i] = [item[0] for item in option]
            # [0] is [CLS], [-1] is sep
            options = torch.tensor(options)
            self.data.append(
                {"article": article, "options": options, "answers": answers}
            )
            self.meta.append(
                {
                    "n_blanks_before": n_blanks_before,
                    "n_blanks_truncated": n_blanks,
                    "article_length": length,
                }
            )

        print("%d answers contains multiple tokens" % (cnt))
        print("%d articles exceeds max length" % (cnt1))
        print("%d answers cannot be decided using the initial token" % (cnt2))

    @staticmethod
    def foo(x):
        x = x.encode("ascii")
        return int.from_bytes(x, byteorder="little")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data_list(split=None, folder=DATASET_ROOT):
    """loads data as nested dicts/lists"""
    lst = []
    for root, dirs, files in os.walk(folder, topdown=False):
        if "ipynb" in root:
            continue  # jupyter tmp file
        for name in sorted(files):
            if split is None or split in root:
                name = os.path.join(root, name)
                with open(name) as f:
                    tmp = json.load(f)
                    lst.append(tmp)
                    if not tmp["options"]:
                        raise
    print(folder, split, len(lst))
    return lst


def collate_fn(data_list):
    batch = {}
    max_len = {}
    for key in data_list[0]:
        max_len[key] = 0
        for item in data_list:
            max_len[key] = max(max_len[key], item[key].shape[0])
        lst = [item[key] for item in data_list]
        padding_value = 0
        if key == "answers":
            padding_value = -1
        batch[key] = pad_sequence(lst, batch_first=True, padding_value=padding_value)
    return batch


def load_cloth_dataset(args, tokenizer, mode="train"):
    assert mode in ["train", "eval", "test"]
    if mode in ["train", "eval"]:
        val_set = ClozeDataset(load_data_list(split="valid"), args.max_seq_length, tokenizer)
        val_loader = DataLoader(
            val_set, 
            batch_size=1, 
            shuffle=False, 
            collate_fn=collate_fn
        )

        if mode == "train":
            train_set = ClozeDataset(load_data_list(split="train"), args.max_seq_length, tokenizer)
            train_loader = DataLoader(
                train_set, 
                batch_size=args.train_batch_size, 
                shuffle=True, 
                collate_fn=collate_fn
            )
            return train_loader, val_loader
        else:
            return val_loader
    else:
        test_set = ClozeDataset(load_data_list(split="test"), args.max_seq_length, tokenizer)
        test_loader = DataLoader(
            test_set, 
            batch_size=1, 
            shuffle=False,
            collate_fn=collate_fn
        )
        return test_loader


""" Model Definition """


def postprocess_predictions(result, article, options, answers=None):
    # we compute our custom loss, so there is no need to set the labels
    _, logit = result[0], result[1]

    b, l, dim = logit.shape
    blank_mask = article == MASK_ID
    blank_mask = blank_mask.unsqueeze(-1).expand(*logit.shape)
    logit = torch.masked_select(logit, blank_mask).view(-1, dim)

    options = options.view(-1)
    mask = options > 0
    options = torch.masked_select(options.view(-1), mask).view(-1, 4)
    # removes the padding options

    if not answers is None:
        answers = answers.view(-1)
        answers = torch.masked_select(answers, answers >= 0)
        # removes the padding answers
        index = answers.long().unsqueeze(1)
        answer_token = torch.gather(options, 1, index).view(-1)
        # shape: (n_blanks)
        CE = nn.CrossEntropyLoss(reduction="none")
        loss = CE(input=logit, target=answer_token)
        return loss
    else:
        option_score = torch.gather(logit, 1, options)
        prediction = torch.argmax(option_score, dim=1).view(-1)
        return prediction


def load_model_and_tokenizer(args):
    is_sparse_model = args.model_name_or_path.startswith("sparse-")

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
    )

    if is_sparse_model:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path[len("sparse-") :],
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        )

    if is_sparse_model:
        model_name = args.model_name_or_path[len("sparse-") :]
        if model_name.startswith("bert"):
            from modeling_bert import BertForMaskedLM

            SparseModelClass = BertForMaskedLM
        else:
            raise ValueError(
                f"Unrecognized model name: {args.model_name_or_path}"
            )
        model = SparseModelClass.from_pretrained(
            model_name, from_tf=False, config=config,
        )
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path, from_tf=False, config=config,
        )
    
    return model, tokenizer


def build_optimizer(args, model):
    NO_DECAY = ["bias", "Norm", "norm"]
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in NO_DECAY)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in NO_DECAY)
            ],
            "weight_decay": 0,
        },
    ]

    if args.weight_decay > 0:
        optimizer = optim.AdamW(param_groups, lr=args.learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    return optimizer


def test(args):
    model, tokenizer = load_model_and_tokenizer(args)
    test_loader = load_cloth_dataset(args, tokenizer, mode="test")

    model = model.cuda().eval()

    results = {}
    for i, data in enumerate(tqdm(test_loader)):
        result = []
        
        article, options = data["article"], data["options"]
        article, options = article.cuda(), options.cuda()
        
        attention_mask = article > 99
        prediction = model(article, attention_mask=attention_mask, labels=article)
        prediction = postprocess_predictions(prediction, article, options)
        
        for j in range(test_loader.dataset.meta[i]["n_blanks_before"]):
            if j < prediction.shape[0]:
                result.append(chr(ord("A") + prediction[j]))
            else:
                result.append("A")
        results["test%04d" % (i + 1)] = result

    with open("results.json", "w") as f:
        json.dump(results, f)


def train(args):
    """training"""
    model, tokenizer = load_model_and_tokenizer(args)
    optimizer = build_optimizer(args, model)
    train_loader, val_loader = load_cloth_dataset(args, tokenizer, mode="train")

    if args.start_epoch > 0:
        state_dict = torch.load(
            Path(args.output_dir) / f"checkpoint_{args.start_epoch}", map_location="cpu"
        )
        model.load_state_dict(state_dict["model_dict"])
    model = nn.DataParallel(model)

    writer = SummaryWriter()

    for epoch in range(args.start_epoch, args.num_train_epochs):
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            article, options, answers = (
                data["article"].cuda(),
                data["options"].cuda(),
                data["answers"].cuda(),
            )
            attention_mask = article > 99
            prediction = model(article, attention_mask=attention_mask, labels=article)
            loss = postprocess_predictions(prediction, article, options, answers=answers)

            loss = loss.mean()
            loss.backward()

            if i % args.grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar(
                    "loss", loss.item(), i * args.train_batch_size + epoch * len(train_loader.dataset)
                )

        model.eval()

        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader):
                article, options, answers = (
                    data["article"].cuda(),
                    data["options"].cuda(),
                    data["answers"].cuda(),
                )
                attention_mask = article > 99
                prediction = model.module(article, attention_mask=attention_mask, labels=article)
                prediction = postprocess_predictions(prediction, article, options)

                answers = answers.view(-1)
                answers = torch.masked_select(answers, answers >= 0)
                correct += (prediction == answers).sum().item()
                total += prediction.shape[0]

            writer.add_scalar("eval_acc", correct / total, epoch + 1)
        print("epoch %d acc: %f" % (epoch + 1, correct / total))
        
        torch.save(
            {
                "model_dict": model.module.state_dict(),
                "optimizer_dict": optimizer.state_dict(),
                "eval_acc": correct / total,
            },
            Path(args.output_dir) / f"checkpoint_{epoch + 1}",
        )


def valid(args):
    model, tokenizer = load_model_and_tokenizer(args)
    val_loader = load_cloth_dataset(args, tokenizer, mode="eval")

    if args.eval_checkpoint is None:
        ckpt_path = Path(args.output_dir) / f"checkpoint_{args.num_train_epochs}"
    else:
        ckpt_path = Path(args.eval_checkpoint)
    
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist."
    print(f"Loading checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict["model_dict"])

    model.cuda().eval()

    correct = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in tqdm(val_loader):
            article, options, answers = (
                data["article"].cuda(),
                data["options"].cuda(),
                data["answers"].cuda(),
            )
            attention_mask = article > 99
            prediction = model(article, attention_mask=attention_mask, labels=article)
            prediction = postprocess_predictions(prediction, article, options)

            answers = answers.view(-1)
            answers = torch.masked_select(answers, answers >= 0)
            correct += (prediction == answers).sum().item()
            total += prediction.shape[0]

    print(f"eval acc: {correct / total:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, 
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--config_name", default=None, type=str, required=False, 
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str, required=False, 
                        help="Pretrained tokenizer name or path if not the same as model_name")
    
    parser.add_argument("--do_train", default=False, action='store_true', required=False)
    parser.add_argument("--do_eval", default=False, action='store_true', required=False)
    parser.add_argument("--do_predict", default=False, action='store_true', required=False)

    parser.add_argument("--train_batch_size", default=3, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=20, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--grad_acc_steps", default=10, type=int)

    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--eval_checkpoint", default=None, type=str, required=False, 
                        help="Explicitly specify the checkpoint to be evaluated")
    parser.add_argument("--output_dir", default="./CKPT/", type=str, required=False)
    args = parser.parse_args()

    if args.eval_checkpoint is not None:
        assert not args.do_train, "args.eval_checkpoint requires args.do_train=False"
        assert args.do_eval, "args.eval_checkpoint requires args.do_eval=True"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.do_train: train(args)
    if args.do_eval: valid(args)
    if args.do_predict: test(args)


if __name__ == "__main__":
    main()
