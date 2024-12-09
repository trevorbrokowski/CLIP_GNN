import os
import json
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import logging
from transformers import CLIPModel, CLIPProcessor

VG_DIR = '/VisualGenome'

####
####
####
####
####
####
####
####
####

######### Datasets for CLIP Baseline #############

class SceneGraphConstructor:
    def __init__(self, processor, clip_model):
        self.processor = processor
        self.clip_model = clip_model

    def construct_text(self, data):
        objects = data['objects']
        relationships = data['relationships']

        descriptions = []
        for obj in objects:
            obj_names = ', '.join(obj['names'])
            attributes = ', '.join(obj.get('attributes', [])) or "None"
            description = f"Object: {obj_names}, Attributes: {attributes}"
            descriptions.append(description)

        predicates = []
        for rel in relationships:
            subj_id = rel['subject_id']
            obj_id = rel['object_id']
            predicate = rel['predicate']

            subj_obj = next((o for o in objects if o['object_id'] == subj_id), None)
            obj_obj = next((o for o in objects if o['object_id'] == obj_id), None)

            if subj_obj and obj_obj:
                subj_name = ', '.join(subj_obj['names'])
                obj_name = ', '.join(obj_obj['names'])
                rel_description = f"{subj_name} {predicate} {obj_name}"
                predicates.append(rel_description)

        # Combine object descriptions and relationships into a single text
        text = '. '.join(descriptions + predicates)
        return text


class SceneGraphDataset(Dataset):
    def __init__(self, 
                 image_ids, 
                 scene_graphs,
                 image_dir, 
                 processor, 
                 clip_model, 
                 image_size=(224, 224)):
       
        self.ids = image_ids
        self.scene_graphs = scene_graphs
        self.scene_graphs_index = {entry['image_id']: entry for entry in self.scene_graphs}

        self.image_dir = image_dir
        self.processor = processor
        self.clip_model = clip_model
        self.graph_constructor = SceneGraphConstructor(processor, clip_model)
        self.image_size = image_size

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = self.ids[idx]
        data = self.scene_graphs_index.get(item)

        image_id = data['image_id']

        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            raise ValueError(f"Image not found: {image_path}")

        image_tensor = self.image_transform(image)
        image_inputs = self.processor(images=image, return_tensors="pt")

        text_data = self.graph_constructor.construct_text(data)
        #print(text_data)
        text_inputs = self.processor(text=text_data, return_tensors="pt", padding=True, truncation=True)

        return image_id, image_inputs, text_inputs

    def get_original_data(self, idx):
        item = self.ids[idx]
        data = self.scene_graphs_index.get(item)
        objects = data['objects']
        relationships = data['relationships']

        original_graph = {
            "nodes": [
                {
                    "id": obj['object_id'],
                    "names": obj['names'],
                    "attributes": obj.get('attributes', []),
                    "bbox": [obj['x'], obj['y'], obj['w'], obj['h']]
                }
                for obj in objects
            ],
            "edges": [
                {
                    "subject_id": rel['subject_id'],
                    "object_id": rel['object_id'],
                    "predicate": rel['predicate']
                }
                for rel in relationships
            ]
        }
        return original_graph
    def input_text(self, text_data):
        return self.processor(text=text_data, return_tensors="pt", padding=True, truncation=True)

class TextEmbeddingModel(nn.Module):
    def __init__(self, input_dim, output_dim, clip_model):
        super(TextEmbeddingModel, self).__init__()
        self.clip_model = clip_model
        self.linear = nn.Linear(input_dim, output_dim)
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.clip_model.get_text_features(**x)
        return F.normalize(self.linear(x), p=2, dim=-1)


class ImageEmbeddingModel(nn.Module):
    def __init__(self, input_dim, output_dim, clip_model):
        super(ImageEmbeddingModel, self).__init__()
        self.clip_model = clip_model
        self.linear = nn.Linear(input_dim, output_dim)
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.clip_model.get_image_features(**x)
        return F.normalize(self.linear(x), p=2, dim=-1)


class ImageTextAlignmentModel(nn.Module):
    def __init__(self, text_model, image_model, embedding_dim):
        super(ImageTextAlignmentModel, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, image_inputs, text_inputs):
        image_embeddings = self.image_model(image_inputs)
        text_embeddings = self.text_model(text_inputs)

        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(image_embeddings, text_embeddings.T)
        return logits

    def get_image_embeddings(self, image_inputs):
        embeddings = self.image_model(image_inputs)
        return F.normalize(embeddings, p=2, dim=-1)

    def get_text_embeddings(self, text_inputs):
        embeddings = self.text_model(text_inputs)
        return F.normalize(embeddings, p=2, dim=-1)


def contrastive_loss(logits, batch_size, temperature=0.1):
    try:
        labels = torch.arange(batch_size, device=logits.device)
        
        if logits.size(0) != batch_size or logits.size(1) != batch_size:
            print(f"Invalid logits shape: {logits.shape} for batch_size: {batch_size}")
            return torch.tensor(0.0, device=logits.device)  # Return zero loss for invalid inputs
        
        if not ((labels >= 0).all() and (labels < logits.size(1)).all()):
            print(f"Invalid labels: {labels}")
            return torch.tensor(0.0, device=logits.device)  # Return zero loss for invalid labels

        logits = logits / temperature

        loss_image_to_text = F.cross_entropy(logits, labels)
        loss_text_to_image = F.cross_entropy(logits.T, labels)

        return (loss_image_to_text + loss_text_to_image) / 2

    except Exception as e:
        print(f"Exception in contrastive_loss: {e}")
def collate_fn(batch):
    try:
        idx = [item[0] for item in batch]
        image_inputs = [item[1] for item in batch]
        text_inputs = [item[2] for item in batch]

        pixel_values = [item['pixel_values'].squeeze(0) for item in image_inputs]
        pixel_values = torch.stack(pixel_values, dim=0)
        image_inputs = {'pixel_values': pixel_values}

        input_ids = [item['input_ids'].squeeze(0) for item in text_inputs]
        attention_mask = [item['attention_mask'].squeeze(0) for item in text_inputs]

        max_length = max([x.size(0) for x in input_ids])

        input_ids_padded = torch.zeros((len(input_ids), max_length), dtype=torch.long)
        attention_mask_padded = torch.zeros((len(attention_mask), max_length), dtype=torch.long)

        for i, (input_id, attention) in enumerate(zip(input_ids, attention_mask)):
            input_ids_padded[i, :input_id.size(0)] = input_id
            attention_mask_padded[i, :attention.size(0)] = attention

        text_inputs = {'input_ids': input_ids_padded, 'attention_mask': attention_mask_padded}

        return idx, image_inputs, text_inputs
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        return None



def save_checkpoint(model, optimizer, epoch, batch_idx, save_path="Baseline.pt"):
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)


def train(model, dataloader, optimizer, device, accumulation_steps=4, save_path="Baseline.pt", log_interval=100):
    model.train()
    total_loss = 0.0
    batch_loss = 0.0
    optimizer.zero_grad()
    scaler = GradScaler()

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch is None:
            continue  

        try:
            image_ids, image_inputs, text_inputs = batch

            image_inputs = {k: v.to(device, non_blocking=True) for k, v in image_inputs.items()}
            text_inputs = {k: v.to(device, non_blocking=True) for k, v in text_inputs.items()}

            with autocast():
                logits = model(image_inputs, text_inputs)
                loss = contrastive_loss(logits, logits.size(0)) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            batch_loss += loss.item() * accumulation_steps

            del image_inputs, text_inputs, logits, loss
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Exception at batch {batch_idx}: {e}")
            traceback.print_exc()
            continue

        if (batch_idx + 1) % log_interval == 0:
            avg_batch_loss = batch_loss / log_interval
            print(f"Batch [{batch_idx + 1}/{len(dataloader)}], Average Loss: {avg_batch_loss:.4f}")
            save_checkpoint(model, optimizer, 0, batch_idx, save_path)
            batch_loss = 0.0

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def load_json(filename):
    with open(os.path.join(VG_DIR, filename), 'r') as f:
        return json.load(f)



####
####
####
####
####
####
####
####
####

######### Getting Main Results CLIP Baseline #############


import torch
from torch.nn.functional import cosine_similarity
import pandas as pd
from tqdm import tqdm

def compute_similarities_with_statistics(graph_text_dict, device, output_prefix, top_k=[1, 5, 10]):
    model = graph_text_dict['model']
    dataloader = graph_text_dict['dataloader']
    model.eval()
    with torch.no_grad():
        image_embeddings_list = []
        graph_embeddings_list = []
        all_idxs = []
        total_similarity = 0
        num_pairs = 0
        similarity_scores = []

        for batch in tqdm(dataloader):
            if batch is None:
                continue  

            try:

                idx, image_inputs, graph_inputs = batch

                all_idxs.extend(idx)

                if output_prefix == 'text':
                    graph_embeddings = model.get_text_embeddings(graph_inputs)
                else:
                    graph_embeddings = model.get_graph_embeddings(graph_inputs)

                image_embeddings = model.get_image_embeddings(image_inputs)

                try:
                    similarity = cosine_similarity(image_embeddings, graph_embeddings, dim=-1)
                except:
                    continue
                total_similarity += similarity.sum().item()
                num_pairs += similarity.size(0)

                similarity_scores.extend(similarity.tolist())

                image_embeddings_list.append(image_embeddings)
                graph_embeddings_list.append(graph_embeddings)
            except Exception as e:
                print(f"An error occurred: {e}")
                continue

        all_image_embeddings = torch.cat(image_embeddings_list, dim=0)
        all_graph_embeddings = torch.cat(graph_embeddings_list, dim=0)

        torch.save(all_image_embeddings, f'image_embeddings_{output_prefix}.pt')
        torch.save(all_graph_embeddings, f'graph_embeddings_{output_prefix}.pt')

        idxs_df = pd.DataFrame({'idx': all_idxs})
        idxs_df.to_csv(f'{output_prefix}_test_idxs.csv', index=False)

        avg_similarity = total_similarity / num_pairs if num_pairs > 0 else 0
        max_similarity = max(similarity_scores) if similarity_scores else 0
        min_similarity = min(similarity_scores) if similarity_scores else 0
        median_similarity = torch.median(torch.tensor(similarity_scores)).item() if similarity_scores else 0
        std_similarity = torch.std(torch.tensor(similarity_scores)).item() if similarity_scores else 0

        print(f"Average image-graph similarity: {avg_similarity}")
        print(f"Max similarity: {max_similarity}")
        print(f"Min similarity: {min_similarity}")
        print(f"Median similarity: {median_similarity}")
        print(f"Standard deviation: {std_similarity}")

        similarity_matrix = torch.matmul(all_image_embeddings, all_graph_embeddings.T)

        num_samples = all_image_embeddings.size(0)
        ranks = torch.zeros(num_samples)
        for i in range(num_samples):
            sims = similarity_matrix[i]
            sorted_indices = torch.argsort(sims, descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            ranks[i] = rank + 1  

        recall_at_k = {}
        for k in top_k:
            recall_at_k[f'Recall@{k}'] = (ranks <= k).float().mean().item()
            print(f"Recall@{k}: {recall_at_k[f'Recall@{k}']:.4f}")

        return {
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "min_similarity": min_similarity,
            "median_similarity": median_similarity,
            "std_similarity": std_similarity,
            "recall_at_k": recall_at_k,
            "similarity_matrix": similarity_matrix
        }



def main():
    image_data = load_json('image_data.json')
    image_data_dict = {item['image_id']: item for item in image_data}
    scene_graphs = load_json('scene_graphs.json')

    image_dir = "VisualGenome/images/all_images"

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)

    embedding_dim = 512  

    text_model = TextEmbeddingModel(
        input_dim=512,
        output_dim=embedding_dim,
        clip_model=clip_model
    ).to(device)

    image_model = ImageEmbeddingModel(
        input_dim=512,
        output_dim=embedding_dim,
        clip_model=clip_model
    ).to(device)

    alignment_model = ImageTextAlignmentModel(
        text_model=text_model,
        image_model=image_model,
        embedding_dim=embedding_dim
    ).to(device)

    optimizer = torch.optim.Adam(alignment_model.parameters(), lr=1e-4)


    image_ids = list(image_data_dict.keys())
    train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

    dataset = SceneGraphDataset(
        image_ids=train_ids,
        scene_graphs=scene_graphs,
        image_dir=image_dir,
        processor=clip_processor,
        clip_model=clip_model
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    test_dataset = SceneGraphDataset(
    image_ids=test_ids,
    scene_graphs=scene_graphs,
    image_dir=image_dir,
    processor=clip_processor,
    clip_model=clip_model
)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    for epoch in range(10):
        loss = train(alignment_model, dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    graph_text_dict = {'model': alignment_model, 
            'dataset': test_dataset, 
            'dataloader': test_dataloader}
    output_prefix = 'text'
    results = compute_similarities_with_statistics(
        graph_text_dict=graph_text_dict,
        device=device,
        output_prefix=output_prefix,
        top_k=[1, 5, 10]
    )





if __name__ == "__main__":
    main()
    

