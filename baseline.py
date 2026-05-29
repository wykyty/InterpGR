from abc import ABC

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import get_constant_schedule
from accelerate import Accelerator
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import json
from utils.io import write_file, read_file
import os
from collections import defaultdict
import wandb


# train 307373 dev 7830
class NQDataset(Dataset, ABC):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        query, doc_id = self.data[item]
        return (torch.tensor(self.tokenizer.encode(str(query), truncation=True, max_length=self.max_len)),
                torch.tensor(self.tokenizer.encode(str(doc_id))))

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data):
        inputs, outputs = zip(*data)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        return {
            'input_ids': inputs,
            'attention_mask': inputs.ne(0),
            'labels': pad_sequence(outputs, batch_first=True, padding_value=-100),
        }


class NewNQDataset(Dataset, ABC):
    def __init__(self, data, corpus, tokenizer, max_len=128):
        self.data = data
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        query, doc_id = self.data[item]
        while isinstance(doc_id, list):
            doc_id = doc_id[0]
        doc = self.corpus[doc_id]
        return (torch.tensor(self.tokenizer.encode(str(query), truncation=True, max_length=self.max_len)),
                torch.tensor(self.tokenizer.encode(str(doc), truncation=True, max_length=self.max_len)))

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data):
        inputs, outputs = zip(*data)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        return {
            'input_ids': inputs,
            'attention_mask': inputs.ne(0),
            'labels': pad_sequence(outputs, batch_first=True, padding_value=-100),
        }


class Tree:
    def __init__(self):
        self.root = dict()

    def set(self, path):
        pointer = self.root
        for i in path:
            if i not in pointer:
                pointer[i] = dict()
            pointer = pointer[i]

    def set_all(self, path_list):
        for path in tqdm(path_list):
            self.set(path)

    def find(self, path):
        if isinstance(path, torch.Tensor):
            path = path.cpu().tolist()
        pointer = self.root
        for i in path:
            if i not in pointer:
                return []
            pointer = pointer[i]
        return list(pointer.keys())

    def __call__(self, batch_id, path):
        res = self.find(path)
        if not res:
            return [1]
        return res


# corpus: "id", 'new_id', '2', "doc", '3', '4', 'en'
# train: "query", "qid", "new_id", "old_id"


def train_atomic():
    accelerator = Accelerator()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    epochs = 100
    batch_size = 64

    if accelerator.is_local_main_process:
        wandb.init(project='dsi-atomic', config={'epochs': epochs, 'batch_size': batch_size, 'lr': 5e-4, 'model': 't5-large', 'num_new_tokens': 109739})
    save_path = 'out/dsi-atomic'
    model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-large')
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-large')

    num_of_new_tokens = 109739

    tokenizer.add_tokens([f'${i}$' for i in range(num_of_new_tokens)])
    model.resize_token_embeddings(len(tokenizer))

    resume_from = 'out/dsi/60.pt'
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        accelerator.print(f'Resuming from {resume_from}')
        model.load_state_dict(torch.load(resume_from, map_location='cpu'))
        start_epoch = int(os.path.basename(resume_from).split('.')[0]) + 1

    data = json.load(open('dataset/nq320k/train.json'))
    # data.extend(json.load(open('dataset/nq320k/qg.json')))

    optimizer = AdamW(model.parameters(), 2e-4)

    dataset = NQDataset(data=data, tokenizer=tokenizer, max_len=32)
    accelerator.print(f'data size={len(dataset)}')
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=True, num_workers=8)

    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    scheduler = get_constant_schedule(optimizer)

    os.makedirs(save_path, exist_ok=True)
    accelerator.print(tokenizer.decode(dataset[128][0]))
    accelerator.print('==>')
    accelerator.print(tokenizer.decode(dataset[128][1]), dataset[128][1])

    for epoch in range(start_epoch, epochs):
        accelerator.print(f'Training epoch {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch in tk0:
            out = model(**batch)
            loss = out.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_report.append(loss.item())
            tk0.set_postfix(loss=sum(loss_report) / len(loss_report))
        avg_loss = sum(loss_report) / len(loss_report)
        if accelerator.is_local_main_process:
            wandb.log({'epoch': epoch, 'loss': avg_loss})
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process and (epoch % 10 == 0 or epoch == epochs - 1):
            accelerator.save(accelerator.unwrap_model(model).state_dict(), f'{save_path}/{epoch}.pt')
    if accelerator.is_local_main_process:
        wandb.finish()


def train_semantic():
    accelerator = Accelerator(mixed_precision="bf16")
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    epochs = 100
    batch_size = 64  # per GPU, total = 64 * 8 = 512
    lr = 5e-4
    save_path = 'out/dsi-semantic-bert'
    resume_from = 'out/dsi-semantic-bert/49.pt'  # 设为 None 则从头训练
    start_epoch = 0

    if accelerator.is_local_main_process:
        wandb.init(
            project='dsi-semantic-bert',
            id='hxgsj3h2',  # 接着继续训练
            resume='allow',
            config={
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'model': 't5-large',
                'num_new_tokens': 30,
            },
        )

    model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-large')
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-large')

    num_of_new_tokens = 30  # 109739

    tokenizer.add_tokens([f'${i}$' for i in range(num_of_new_tokens)])
    model.resize_token_embeddings(len(tokenizer))

    if resume_from and os.path.exists(resume_from):
        accelerator.print(f'Resuming from {resume_from}')
        model.load_state_dict(torch.load(resume_from, map_location='cpu'))
        start_epoch = int(os.path.basename(resume_from).split('.')[0]) + 1

    data = json.load(open('dataset/nq320k/train.json'))
    # data.extend(json.load(open('dataset/nq320k/qg.json')))

    corpus = json.load(open('dataset/nq320k_id/id.semantic.bert.json'))
    corpus = [''.join([f'${i}$' for i in z]) for z in corpus]

    optimizer = AdamW(model.parameters(), lr)

    dataset = NewNQDataset(data=data, corpus=corpus, tokenizer=tokenizer, max_len=32)
    accelerator.print(f'data size={len(dataset)}')
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=True, num_workers=8)

    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    scheduler = get_constant_schedule(optimizer)

    os.makedirs(save_path, exist_ok=True)
    accelerator.print(tokenizer.decode(dataset[128][0]))
    accelerator.print('==>')
    accelerator.print(tokenizer.decode(dataset[128][1]), dataset[128][1])

    global_step = 0
    for epoch in range(start_epoch, epochs):
        accelerator.print(f'Training epoch {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        epoch_losses = []
        for batch in tk0:
            out = model(**batch)
            loss = out.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            tk0.set_postfix(loss=avg_loss)

            if accelerator.is_local_main_process and global_step % 100 == 0:
                wandb.log({'step': global_step, 'batch_loss': loss_val, 'avg_loss': avg_loss})

            global_step += 1

        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        if accelerator.is_local_main_process:
            wandb.log({'epoch': epoch, 'epoch_loss': epoch_avg_loss})
            if epoch % 10 == 0 or epoch == epochs - 1:
                accelerator.save(
                    accelerator.unwrap_model(model).state_dict(),
                    f'{save_path}/{epoch}.pt'
                )

    if accelerator.is_local_main_process:
        wandb.finish()


def test_atomic():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    batch_size = 32
    save_path = 'out/dsi'
    num_of_new_tokens = 109739

    model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-large')
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-large')

    tokenizer.add_tokens([f'${i}$' for i in range(num_of_new_tokens)])
    model.resize_token_embeddings(len(tokenizer))

    data = json.load(open('dataset/nq320k/dev.json'))

    dataset = NQDataset(data=data, tokenizer=tokenizer, max_len=32)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, shuffle=False, num_workers=8)
    model = model.cuda()
    model.eval()

    available = sorted([int(f.split('.')[0]) for f in os.listdir(save_path) if f.endswith('.pt') and f.split('.')[0].isdigit()])
    if not available:
        print("No checkpoint found!")
        return
    epoch = 70
    print(f'Loading checkpoint {save_path}/{epoch}.pt')
    model.load_state_dict(torch.load(f'{save_path}/{epoch}.pt', map_location='cuda'))

    tk0 = tqdm(data_loader, total=len(data_loader))
    top_k = 10
    hit1, hit10 = [], []
    with torch.no_grad():
        for batch in tk0:
            batch = {k: v.cuda() for k, v in batch.items()}
            output = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=10,
                num_beams=top_k,
                num_return_sequences=top_k,
            )
            output = tokenizer.batch_decode(output, skip_special_tokens=True)
            output = [str(x).replace('$', '').strip() for x in output]
            # Group top-k outputs per query
            preds = [output[i:i+top_k] for i in range(0, len(output), top_k)]

            batch['labels'][batch['labels'] == -100] = 0
            labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            labels = [str(x).replace('$', '').strip() for x in labels]

            hit1.extend([int(p[0] == l) for p, l in zip(preds, labels)])
            hit10.extend([int(l in p) for p, l in zip(preds, labels)])
            tk0.set_postfix(hit1=sum(hit1)/len(hit1), hit10=sum(hit10)/len(hit10))

    print(f'Epoch {epoch}, Hit@1 = {sum(hit1)/len(hit1):.4f}, Hit@10 = {sum(hit10)/len(hit10):.4f}')


def test():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    batch_size = 1
    save_path = 'out/dsi-ms-title'
    # save_path = 'out/dsi-title'
    num_of_new_tokens = 10  # 109739

    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
    tokenizer = AutoTokenizer.from_pretrained('t5-base')


    # tokenizer.add_tokens([f'${i}$' for i in range(num_of_new_tokens)])  # 109739
    # model.resize_token_embeddings(len(tokenizer))

    # tokenizer = AutoTokenizer.from_pretrained("./genre-kilt")
    # model = AutoModelForSeq2SeqLM.from_pretrained("./genre-kilt").eval()

    data = json.load(open('data/new_nq320k/dev_unseen.json'))
    corpus = json.load(open('data/new_nq320k/id.newtitle.json'))

    qq = read_file('out/flan-t5-xxl/nq320k.title')
    print(len(data), len(qq))
    data = [[_q.lower(), _x[1]] for _x, _q in zip(data, qq)]

    # corpus = json.load(open('data/new_nq320k/id.bert_km.json'))
    # corpus = [''.join([f'${i}$' for i in z]) for z in corpus]

    # data = [[doc, i] for i, doc in enumerate(read_file('data/ms320k/corpus.txt'))]
    # corpus = ['' for _ in range(len(data))]

    # corpus = [f'${z}$' for z in range(109739)]

    # data = json.load(open('data/ms320k/new_dev.json'))
    # corpus = read_file('data/ms320k/id.title.txt')
    # corpus = json.load(open('data/ms320k/id.semantic.json'))
    # corpus = [''.join([f'${i}$' for i in z]) for z in corpus]

    # corpus = json.load(open('data/new_nq320k/id.newtitle.json'))

    # from run_bi import load_beir
    # from collections import defaultdict
    # data, corpus = load_beir('scidocs')
    # corpus = [' '.join(x.replace('Title: ', '').replace('. Text:', '').strip().split()[:8]).lower() for x in corpus]
    # docid_to_doc = defaultdict(list)
    # for i, item in enumerate(corpus):
    #     docid_to_doc[item].append(i)
    # query_ids = [x[1] for x in data]

    print(len(data), len(corpus))

    corpus_ids = [[0] + tokenizer.encode(line) for line in corpus]
    print(corpus_ids[0])
    tree = Tree()
    tree.set_all(corpus_ids)

    dataset = NewNQDataset(data=data, corpus=corpus, tokenizer=tokenizer, max_len=128)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=False, num_workers=16)
    model = model.cuda()
    model.eval()
    seen_split = json.load(open('data/new_nq320k/dev_seen_split.json'))
    unseen_split = json.load(open('data/new_nq320k/dev_unseen_split.json'))
    for epoch in range(10000, 0, -1):
        if not os.path.exists(f'{save_path}/{epoch}.pt'):
            continue
        # print(f'Test {save_path}/{epoch}.pt')
        # model.load_state_dict(torch.load(f'{save_path}/{epoch}.pt'))
        tk0 = tqdm(data_loader, total=len(data_loader))
        acc = []
        output_all = []
        label_all = []
        top_k = 1
        with torch.no_grad():
            for batch in tk0:
                batch = {k: v.cuda() for k, v in batch.items()}
                output = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=4,
                    num_beams=top_k,
                    num_return_sequences=top_k,
                    length_penalty=None,
                    min_length=None,
                    no_repeat_ngram_size=None,
                    early_stopping=None,
                    prefix_allowed_tokens_fn=tree
                )
                # continue
                output = tokenizer.batch_decode(output, skip_special_tokens=True)
                output = [str(x).replace('$', '').strip() for x in output]
                # print(output)
                beam = []
                new_output = []
                for line in output:
                    if len(beam) >= top_k:
                        new_output.append(beam)
                        beam = []
                    beam.append(line)
                new_output.append(beam)
                # print(len(output))
                batch['labels'][batch['labels'] == -100] = 0
                labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                labels = [str(x).replace('$', '').strip() for x in labels]

                acc.extend([int(l in o) for o, l in zip(new_output, labels)])
                tk0.set_postfix(acc=sum(acc) / len(acc))

                # print(new_output[-1], labels[-1])

                print(new_output)
                print(labels)

                output_all.extend(new_output)
                label_all.extend(labels)
        print(f'Test {save_path}/{epoch}.pt, ACC =', sum(acc) / len(acc), end='; ')
        # print('Seen', sum([acc[j] for j in seen_split]) / len(seen_split), end='; ')
        # print('Unseen', sum([acc[j] for j in unseen_split]) / len(unseen_split))
        json.dump([output_all, label_all], open(f'{save_path}/{epoch}.pt.outputs', 'w'))
        from eval import eval_all
        print(eval_all(output_all, label_all))

        break

        # new_predictions = []
        # import copy
        # for line in output_all:
        #     new_line = []
        #     for s in line:
        #         if s not in docid_to_doc:
        #             continue
        #         tmp = copy.deepcopy(docid_to_doc[s])
        #         new_line.extend(tmp)
        #         if len(new_line) > 10:
        #             break
        #     new_predictions.append(new_line)
        # output_all = new_predictions
        #
        # print('BEIR', eval_all(output_all, query_ids))

        # print('Seen')
        # print(eval_all([output_all[j] for j in seen_split], [label_all[j] for j in seen_split]))
        # print('Unseen')
        # print(eval_all([output_all[j] for j in unseen_split], [label_all[j] for j in unseen_split]))


def simple_match():
    import edlib
    import numpy as np
    from thefuzz import fuzz
    data = json.load(open('data/new_nq320k/dev_unseen.json'))
    corpus = json.load(open('data/new_nq320k/id.newtitle.json'))

    # qq = read_file('out/flan-t5-xxl/nq320k.title')
    qq = read_file('out/code-002/nq320k.title')
    output_all = []
    label_all = []
    metric = []
    for line, item in zip(tqdm(qq), data):
        score = [fuzz.token_sort_ratio(line.lower(), x) for x in corpus]
        # score = [- edlib.align(line.lower(), x)['editDistance'] for x in corpus]
        idx = np.argmax(score)
        output_all.append(corpus[idx])
        label_all.append(corpus[item[1]])
        print(line, corpus[idx], corpus[item[1]], sep=' | ')
        metric.append(idx == item[1])
        print(sum(metric) / len(metric))



def simple_test():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    batch_size = 10

    model = AutoModelForSeq2SeqLM.from_pretrained('./parrot_paraphraser_t5')
    tokenizer = AutoTokenizer.from_pretrained('./parrot_paraphraser_t5')

    data = json.load(open('data/new_nq320k/dev_unseen.json'))
    corpus = json.load(open('data/new_nq320k/id.newtitle.json'))

    # qq = read_file('out/flan-t5-xxl/nq320k.title')
    qq = read_file('out/code-002/nq320k.title')
    print(len(data), len(qq))
    data = [[_q.lower(), _x[1]] for _x, _q in zip(data, qq)]

    print(len(data), len(corpus))

    corpus_ids = [[0] + tokenizer.encode(line) for line in corpus]
    # corpus_ids = [[2, 0] + tokenizer.encode(line)[1:] for line in corpus]
    print(corpus_ids[0])
    tree = Tree()
    tree.set_all(corpus_ids)

    dataset = NewNQDataset(data=data, corpus=corpus, tokenizer=tokenizer, max_len=128)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=False, num_workers=16)
    model = model.cuda()
    model.eval()
    seen_split = json.load(open('data/new_nq320k/dev_seen_split.json'))
    unseen_split = json.load(open('data/new_nq320k/dev_unseen_split.json'))
    for epoch in range(0, 100):
        # print(f'Test {save_path}/{epoch}.pt')
        # model.load_state_dict(torch.load(f'{save_path}/{epoch}.pt'))
        tk0 = tqdm(data_loader, total=len(data_loader))
        acc = []
        output_all = []
        label_all = []
        top_k = 32
        with torch.no_grad():
            for batch in tk0:
                batch = {k: v.cuda() for k, v in batch.items()}
                output = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=32,
                    num_beams=top_k,
                    num_return_sequences=1,
                    length_penalty=0,
                    no_repeat_ngram_size=0,
                    early_stopping=False,
                    prefix_allowed_tokens_fn=tree,

                    min_length=3,

                    # bos_token_id=0,
                    # decoder_start_token_id=2,
                    # eos_token_id=2,
                    # forced_bos_token_id=0,
                    # forced_eos_token_id=2,
                )
                # print(output)
                # continue
                output = tokenizer.batch_decode(output, skip_special_tokens=True)
                output = [str(x).replace('$', '').strip() for x in output]
                # print(output)
                beam = []
                new_output = []
                for line in output:
                    if len(beam) >= top_k:
                        new_output.append(beam)
                        beam = []
                    beam.append(line)
                new_output.append(beam)
                # print(len(output))
                batch['labels'][batch['labels'] == -100] = 0
                labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                labels = [str(x).replace('$', '').strip() for x in labels]

                acc.extend([int(l in o) for o, l in zip(new_output, labels)])
                tk0.set_postfix(acc=sum(acc) / len(acc))

                # print(new_output[-1], labels[-1])

                print(tokenizer.batch_decode(batch['input_ids'],skip_special_tokens=True), new_output, labels)
                # print(labels)

                output_all.extend(new_output)
                label_all.extend(labels)

        print(f'Test {epoch}.pt, ACC =', sum(acc) / len(acc), end='; ')
        # print('Seen', sum([acc[j] for j in seen_split]) / len(seen_split), end='; ')
        # print('Unseen', sum([acc[j] for j in unseen_split]) / len(unseen_split))
        json.dump([output_all, label_all], open(f'{save_path}/{epoch}.pt.outputs', 'w'))
        from eval import eval_all
        print(eval_all(output_all, label_all))

        break

        # new_predictions = []
        # import copy
        # for line in output_all:
        #     new_line = []
        #     for s in line:
        #         if s not in docid_to_doc:
        #             continue
        #         tmp = copy.deepcopy(docid_to_doc[s])
        #         new_line.extend(tmp)
        #         if len(new_line) > 10:
        #             break
        #     new_predictions.append(new_line)
        # output_all = new_predictions
        #
        # print('BEIR', eval_all(output_all, query_ids))

        # print('Seen')
        # print(eval_all([output_all[j] for j in seen_split], [label_all[j] for j in seen_split]))
        # print('Unseen')
        # print(eval_all([output_all[j] for j in unseen_split], [label_all[j] for j in unseen_split]))


# corpus: "id", '1', '2', "doc", '3', '4', 'en'
# train: "query", "qid", "new_id", "old_id"

def do():
    train_data = json.load(open('data/new_nq320k/train.json'))
    dev_data = json.load(open('data/new_nq320k/dev.json'))
    qg = json.load(open('data/new_nq320k/qg.json'))
    train_id = set([x[1] for x in train_data])
    dev_id = set([x[1] for x in train_data])

    # new_data = [x for x in dev_data if x[1] not in train_id]
    # new_data = [x for x in qg if x[1] not in train_id]
    new_data = [i for i, x in enumerate(dev_data) if x[1] not in train_id]
    # print(len(new_data))
    print(len(new_data))
    # json.dump(new_data, open('data/new_nq320k/dev_seen_split.json', 'w'))  # 6075
    json.dump(new_data, open('data/new_nq320k/dev_unseen_split.json', 'w'))  # 1755

    # json.dump(new_data, open('data/new_nq320k/qg_seen.json', 'w'))  # 1080260
    # json.dump(new_data, open('data/new_nq320k/qg_unseen.json', 'w'))  # 17130


def title_data():
    # "query", "qid", "new_id", "old_id"
    # corpus: "id", '1', '2', "doc", '3', '4', 'en'
    data = [line[:-1].split('\t') for line in open('data/nq320k/train.txt')]
    corpus = {}
    for line in [line[:-1].split('\t') for line in open('data/nq320k/corpus.txt')]:
        old_id, _, _, doc, _, _, _ = line
        corpus[old_id] = doc
    new_data = []
    for line in data:
        query, _, _, old_id = line
        doc = corpus[old_id]
        title = doc.split('  ')[0]
        title = ' '.join(title.split()[:5])
        line.append(title)
        new_data.append('\t'.join(line))
    write_file(new_data, 'data/nq320k/train_title.txt')


def clean_data():
    train_data = [line[:-1].split('\t') for line in open('data/nq_data_sem/nq_train_doc_newid.tsv')]
    old_to_new = dict()
    new_train_data = []
    for line in train_data:
        query, qid, title, new_id, old_id = line
        old_to_new[old_id] = new_id
        new_train_data.append([query, int(old_id)])
    dev_data = [line[:-1].split('\t') for line in open('data/nq_data_sem/nq_dev_doc_newid.tsv')]
    new_dev_data = []
    for line in dev_data:
        query, qid, title, new_id, old_id = line
        old_to_new[old_id] = new_id
        new_dev_data.append([query, int(old_id)])

    corpus = [line[:-1].split('\t') for line in open('data/nq320k/corpus.txt')]
    new_corpus = []
    bert_km = []
    random_id = []
    title_id = []
    for line in corpus:
        old_id, _, _, doc, *_ = line
        # title = doc.split('  ')[0]
        # title = ' '.join(title.split()[:5])
        bert_km.append(str(old_to_new[old_id]))
        # random_id.append(str(old_id).zfill(6))
        # title_id.append(title)
        new_corpus.append(doc)
    # json.dump(new_train_data, open('data/new_nq320k/train.json', 'w'))
    # json.dump(new_dev_data, open('data/new_nq320k/dev.json', 'w'))
    json.dump(new_corpus, open('data/new_nq320k/corpus.json', 'w'))
    # json.dump(bert_km, open('data/new_nq320k/id.simcse.json', 'w'))
    # json.dump(random_id, open('data/new_nq320k/id.random.json', 'w'))
    # json.dump(title_id, open('data/new_nq320k/id.title.json', 'w'))


def tmp():
    from run_bi import load_beir
    from collections import defaultdict
    data, corpus = load_beir('scidocs')
    corpus = [' '.join(x.replace('Title: ', '').replace('. Text:', '').strip().split()[:8]).lower() for x in corpus]
    docid_to_doc = defaultdict(list)
    for i, item in enumerate(corpus):
        docid_to_doc[item].append(i)
    query_ids = [x[1] for x in data]

    from eval import eval_all
    import numpy as np
    prediction = []
    
    for i in range(len(data)):
        rank = [m for m in range(len(corpus))]
        # np.random.shuffle(rank)
        prediction.append(rank)
    print(eval_all(prediction, query_ids))




def test_semantic():   # 受限解码出问题了，暂时跑不通
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    batch_size = 32
    save_path = 'out/dsi-semantic-bert'
    num_of_new_tokens = 30
    top_k = 10

    model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-large')
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-large')

    tokenizer.add_tokens([f'${i}$' for i in range(num_of_new_tokens)])
    model.resize_token_embeddings(len(tokenizer))

    corpus = json.load(open('dataset/nq320k_id/id.semantic.bert.json'))
    corpus = [''.join([f'${i}$' for i in z]) for z in corpus]

    # stripped_id_string → list of doc indices
    id_to_docs = defaultdict(list)
    for doc_idx, id_str in enumerate(corpus):
        clean = id_str.replace('$', '')
        id_to_docs[clean].append(doc_idx)

    corpus_ids = [[0] + tokenizer.encode(s) for s in tqdm(corpus, desc="Building tree")]
    tree = Tree()
    tree.set_all(corpus_ids)

    data = json.load(open('dataset/nq320k/dev.json'))
    dataset = NewNQDataset(data=data, corpus=corpus, tokenizer=tokenizer, max_len=32)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, 
                                              shuffle=False, num_workers=4)

    if not os.path.exists(save_path):
        print(f"ERROR: {save_path} not found!")
        return

    # all_ckpts = sorted([
    #     int(f.split('.')[0])
    #     for f in os.listdir(save_path)
    #     if f.endswith('.pt') and f.split('.')[0].isdigit()
    # ])

    # if not all_ckpts:
    #     print(f"ERROR: No checkpoints found in {save_path}!")
    #     return

    checkpoints = [49]
    print(f"Checkpoints to evaluate: {checkpoints}")

    model = model.cuda()
    model.eval()
    for epoch in checkpoints:
        ckpt_path = f'{save_path}/{epoch}.pt'
        print(f'\n=== Epoch {epoch} ({ckpt_path}) ===')
        model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))

        tk0 = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}")
        hit1, hit10 = [], []
        with torch.no_grad():
            for batch in tk0:
                batch = {k: v.cuda() for k, v in batch.items() if v is not None}

                output = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=20,
                    num_beams=top_k,
                    num_return_sequences=top_k,
                    prefix_allowed_tokens_fn=tree,
                )

                raw_decoded = tokenizer.batch_decode(output, skip_special_tokens=False)

                decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
                decoded = [x.replace('$', '').strip() for x in decoded]
                preds = [decoded[i:i + top_k] for i in range(0, len(decoded), top_k)]

                # print("\n" + "="*50)
                # print(f"[DEBUG] 输入 Query 还原: {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)}")
                # print(f"[DEBUG] 模型原始输出 Token IDs: {output[0].cpu().tolist()}")
                # print(f"[DEBUG] 模型原始输出 字符串: {raw_decoded[0]}")
                # print("="*50 + "\n")

                batch['labels'][batch['labels'] == -100] = 0
                labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                labels = [x.replace('$', '').strip() for x in labels]

                for pred_list, label in zip(preds, labels):
                    h1 = label in id_to_docs.get(pred_list[0], [])
                    hit1.append(int(h1))
                    h10 = any(label in id_to_docs.get(p, []) for p in pred_list)
                    hit10.append(int(h10))

        h1 = sum(hit1) / len(hit1)
        h10 = sum(hit10) / len(hit10)
        print(f"Epoch {epoch}: Hits@1={h1:.4f}  Hits@10={h10:.4f}")

        try:
            import wandb as _w
            if _w.run is not None:
                _w.log({'eval/hits@1': h1, 'eval/hits@10': h10, 'eval/epoch': epoch})
        except:
            pass

    print("\nDone.")


def test_semantic2(eval_all_checkpoints=False):  # 可以跑通
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    batch_size = 16   
    save_path = 'out/dsi-semantic-bert'
    num_of_new_tokens = 30
    top_k = 10

    model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-large')
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-large')
    tokenizer.add_tokens([f'${i}$' for i in range(num_of_new_tokens)])
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()
    model.eval()

    raw_dev_data = json.load(open('dataset/nq320k/dev.json'))
    
    semantic_ids = json.load(open('dataset/nq320k_id/id.semantic.bert.json'))
    corpus_strs = [''.join([f'${i}$' for i in z]) for z in semantic_ids]

    # 建立从 "纯Token ID元组" 到 "文档索引" 的双向映射
    tuple_to_docs = defaultdict(list)
    doc_to_tuple = {}
    
    for doc_idx, z in enumerate(semantic_ids):
        # z 是类似于 [14, 7, 19, 2] 的数字列表
        # 我们把它还原为模型实际训练时的真实 Token ID 序列
        token_ids = []
        for token_val in z:
            # 找到 f"${token_val}$" 在当前 tokenizer 里的真实 ID
            t_id = tokenizer.convert_tokens_to_ids(f'${token_val}$')
            token_ids.append(t_id)
        
        token_tuple = tuple(token_ids) # 转换为不可变的 tuple 作为字典键
        tuple_to_docs[token_tuple].append(doc_idx)
        doc_to_tuple[doc_idx] = token_tuple

    # 3. 规范树的构建
    corpus_token_ids = [[0] + list(doc_to_tuple[idx]) + [1] for idx in range(len(corpus_strs))]
    tree = Tree()
    tree.set_all(corpus_token_ids)

    dataset = NewNQDataset(data=raw_dev_data, corpus=corpus_strs, tokenizer=tokenizer, max_len=32)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
        shuffle=False, num_workers=4,
    )

    all_ckpts = sorted([int(f.split('.')[0]) for f in os.listdir(save_path) if f.endswith('.pt') and f.split('.')[0].isdigit()])
    if not all_ckpts: return
    checkpoints = [40, 49]

    for epoch in checkpoints:
        ckpt_path = f'{save_path}/{epoch}.pt'
        print(f'\n=== Epoch {epoch} ({ckpt_path}) ===')
        model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))

        hit1, hit10 = [], []
        data_ptr = 0 

        with torch.no_grad():
            for batch in tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}"):
                batch_size_actual = batch['input_ids'].size(0)
                batch = {k: v.cuda() for k, v in batch.items() if v is not None}

                output = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=20, 
                    num_beams=top_k,
                    num_return_sequences=top_k,
                    prefix_allowed_tokens_fn=tree,
                )

                # output 形状: [batch_size * top_k, seq_len]
                output = output.cpu().tolist()

                # 提取模型生成的纯有效新 Token 序列
                cleaned_preds = []
                for seq in output:
                    # 过滤掉 T5 的控制符：<pad>(0), </s>(1), <unk>(2)
                    valid_tokens = [t for t in seq if t not in [0, 1, 2]]
                    cleaned_preds.append(tuple(valid_tokens))

                # 按 top_k 分组
                preds = [cleaned_preds[i:i + top_k] for i in range(0, len(cleaned_preds), top_k)]

                for b_idx in range(batch_size_actual):
                    pred_list = preds[b_idx] # 包含了 top_k 个预测 tuple
                    
                    # 拿到当前样本真实的 doc_idx (或是原本的ID)
                    _, true_doc_id = raw_dev_data[data_ptr]
                    while isinstance(true_doc_id, list):
                        true_doc_id = true_doc_id[0]
                    
                    # 拿到这行真实文档对应的标准语义 Token 元组
                    true_token_tuple = doc_to_tuple.get(true_doc_id, None)

                    # 判定 Hit@1: 最优预测的元组是否和标签元组完全一致
                    h1 = (pred_list[0] == true_token_tuple)
                    hit1.append(int(h1))

                    # 判定 Hit@10: 真实的元组是否在预测的候选集里
                    h10 = (true_token_tuple in pred_list)
                    hit10.append(int(h10))

                    data_ptr += 1

        h1 = sum(hit1) / len(hit1)
        h10 = sum(hit10) / len(hit10)
        print(f"👉 修复后结果 -> Epoch {epoch}: Hits@1={h1:.4f}  Hits@10={h10:.4f}")

    print("\nDone.")

def test_semantic3(eval_all_checkpoints=False):  # 可以跑通，topk 扩展到100
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    batch_size = 16  # top_k放大到100，beam search会消耗更多显存，将batch_size稍微调小（如16或32）以防OOM  
    save_path = 'out/dsi-semantic-bert'
    num_of_new_tokens = 30
    top_k = 100  

    model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-large')
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-large')
    tokenizer.add_tokens([f'${i}$' for i in range(num_of_new_tokens)])
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()
    model.eval()

    # 1. 读取原始的 dev.json
    raw_dev_data = json.load(open('dataset/nq320k/dev.json'))
    
    # 2. 读取语义编码，将其直接转化为严格的特殊 Token ID 元组 (Tuple)
    semantic_ids = json.load(open('dataset/nq320k_id/id.semantic.bert.json'))
    corpus_strs = [''.join([f'${i}$' for i in z]) for z in semantic_ids]

    # 建立从 "纯Token ID元组" 到 "文档索引" 的双向映射
    tuple_to_docs = defaultdict(list)
    doc_to_tuple = {}
    
    for doc_idx, z in enumerate(semantic_ids):
        token_ids = []
        for token_val in z:
            t_id = tokenizer.convert_tokens_to_ids(f'${token_val}$')
            token_ids.append(t_id)
        
        token_tuple = tuple(token_ids)
        tuple_to_docs[token_tuple].append(doc_idx)
        doc_to_tuple[doc_idx] = token_tuple

    # 3. 规范树的构建
    corpus_token_ids = [[0] + list(doc_to_tuple[idx]) + [1] for idx in range(len(corpus_strs))]
    tree = Tree()
    tree.set_all(corpus_token_ids)

    dataset = NewNQDataset(data=raw_dev_data, corpus=corpus_strs, tokenizer=tokenizer, max_len=32)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
        shuffle=False, num_workers=4,
    )

    all_ckpts = sorted([int(f.split('.')[0]) for f in os.listdir(save_path) if f.endswith('.pt') and f.split('.')[0].isdigit()])
    if not all_ckpts: return
    # checkpoints = all_ckpts if eval_all_checkpoints else [all_ckpts[-1]]
    checkpoints = [30, 40]

    for epoch in checkpoints:
        ckpt_path = f'{save_path}/{epoch}.pt'
        print(f'\n=== Epoch {epoch} ({ckpt_path}) ===')
        model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))

        # 🚨 丰富指标统计列表
        hit1, hit10, hit100, mrr_list = [], [], [], []
        data_ptr = 0 

        with torch.no_grad():
            for batch in tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}"):
                batch_size_actual = batch['input_ids'].size(0)
                batch = {k: v.cuda() for k, v in batch.items() if v is not None}

                output = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=15, 
                    num_beams=top_k,
                    num_return_sequences=top_k,
                    prefix_allowed_tokens_fn=tree,
                )

                raw_decoded = tokenizer.batch_decode(output, skip_special_tokens=False)

                decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
                decoded = [x.replace('$', '').strip() for x in decoded]
                preds = [decoded[i:i + top_k] for i in range(0, len(decoded), top_k)]

                # print("\n" + "="*50)
                # print(f"[DEBUG] 输入 Query 还原: {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)}")
                # print(f"[DEBUG] 模型原始输出 Token IDs: {output[0].cpu().tolist()}")
                # print(f"[DEBUG] 模型原始输出 字符串: {raw_decoded[0]}")
                # print("="*50 + "\n")

                output = output.cpu().tolist()

                # 提取模型生成的纯有效新 Token 序列
                cleaned_preds = []
                for seq in output:
                    valid_tokens = [t for t in seq if t not in [0, 1, 2]]
                    cleaned_preds.append(tuple(valid_tokens))

                # 按 top_k 分组
                preds = [cleaned_preds[i:i + top_k] for i in range(0, len(cleaned_preds), top_k)]

                for b_idx in range(batch_size_actual):
                    pred_list = preds[b_idx]  # 包含了 100 个候选预测元组
                    
                    # 拿到当前样本真实的 doc_idx
                    _, true_doc_id = raw_dev_data[data_ptr]
                    while isinstance(true_doc_id, list):
                        true_doc_id = true_doc_id[0]
                    
                    # 拿到这行真实文档对应的标准语义 Token 元组
                    true_token_tuple = doc_to_tuple.get(true_doc_id, None)

                    # ─── 1. 计算 Hits@K ───
                    hit1.append(int(pred_list[0] == true_token_tuple))
                    hit10.append(int(true_token_tuple in pred_list[:10]))
                    hit100.append(int(true_token_tuple in pred_list[:100]))

                    # ─── 2. 计算 MRR (Mean Reciprocal Rank) ───
                    rr = 0.0  # Reciprocal Rank 默认为 0
                    if true_token_tuple in pred_list:
                        # 找到第一个命中的位置（0-indexed 索引，所以计算排名时要 +1）
                        rank = pred_list.index(true_token_tuple) + 1
                        rr = 1.0 / rank
                    mrr_list.append(rr)

                    data_ptr += 1

        # ─── 打印最终报表 ───
        h1 = sum(hit1) / len(hit1)
        h10 = sum(hit10) / len(hit10)
        h100 = sum(hit100) / len(hit100)
        mrr = sum(mrr_list) / len(mrr_list)
        
        print(f"\n================📊 EPOCH {epoch} EVAL REPORT ================")
        print(f"Hits@1   : {h1:.4f}")
        print(f"Hits@10  : {h10:.4f}")
        print(f"Hits@100 : {h100:.4f}")
        print(f"MRR      : {mrr:.4f}")
        print(f"============================================================")

        try:
            import wandb as _w
            if _w.run is not None:
                _w.log({
                    'eval/hits@1': h1, 
                    'eval/hits@10': h10, 
                    'eval/hits@100': h100, 
                    'eval/mrr': mrr, 
                    'eval/epoch': epoch
                })
        except:
            pass

    print("\nDone.")

if __name__ == '__main__':
    # train()
    # test_atomic()

    train_semantic()
    # test_semantic3()
    
    # tmp()
    # exit()
    # # clean_data()
    # # do()
    # # train()
    # test()
    # simple_test()
    # simple_match()
    # while True:
    #     test()
    # title_data()

