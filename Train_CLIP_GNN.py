import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch as GraphBatch
from torch_geometric.nn import GATConv, global_mean_pool, TopKPooling
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.nn import GCNConv
from transformers import CLIPModel, CLIPProcessor
import os

import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import traceback

import torch.nn as nn
from torch.nn.functional import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
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

######### Dataset for CLIP_GNN #############

class GraphClipDataset(Dataset):
    def __init__(self, 
                image_ids, 
                processor, 
                clip_model,
                image_data_dict,
                objects_data_dict,
                attributes_data_dict,
                relationships_data_dict,
                synsets_data_dict,
                ):
                self.ids = image_ids
                self.processor = processor
                self.clip_model = clip_model
                self.images = image_data_dict
                self.objects = objects_data_dict
                self.attributes = attributes_data_dict
                self.relationships = relationships_data_dict
                self.synsets = synsets_data_dict


    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        item = self.ids[idx]

        image_url = f'VisualGenome/images/all_images/{item}.jpg'
        image = Image.open(image_url).convert('RGB')

        image_inputs = self.processor(images=image, return_tensors="pt")

        node_texts, edge_texts, edge_indices = self._construct_graph_text(item)
        node_embeddings = self._encode_texts(node_texts)
        edge_embeddings = self._encode_texts(edge_texts)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() 
        graph_data = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_embeddings)

        return item, image_inputs, graph_data

    def _construct_graph_text(self, image_id):
       
        objects = self.objects.get(image_id, [])
        attributes = self.attributes.get(image_id, [])
        relationships = self.relationships.get(image_id, [])
        
        node_texts = [] 
        edge_texts = []
        object_id_to_text = {}

        for obj in objects:
            obj_id = obj['object_id']
            obj_names = ', '.join(obj['names'])
     
            if len(obj['synsets']) > 0:
                obj_synsets = ', '.join([self.synsets.get(synset, '') for synset in obj['synsets']])
            else:
                obj_synsets = 'None'
            
            obj_attributes = next((attr.get('attributes', []) for attr in attributes if attr['object_id'] == obj_id), [])
            attr_text = ', '.join(obj_attributes) if obj_attributes else 'None'
            
            node_text = f"Object: {obj_names},  Attributes: {attr_text}"
            object_id_to_text[obj_id] = node_text
            node_texts.append(node_text)

        edge_indices = []  
        for rel in relationships:
            subj_id = rel['subject']['object_id']
            obj_id = rel['object']['object_id']
            predicate = rel['predicate'] 

            edge_texts.append(predicate)  

            subj_idx = node_texts.index(object_id_to_text[subj_id])
            obj_idx = node_texts.index(object_id_to_text[obj_id])
            edge_indices.append([subj_idx, obj_idx])

        #print(node_texts)
        #print(edge_texts)

        return node_texts, edge_texts, edge_indices

    def _encode_texts(self, text_list):
        if not text_list:
            return torch.zeros(0,512).to(self.clip_model.device)
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(self.clip_model.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**inputs)
        return text_embeddings

    def create_graph_from_dict(self, input_dict):
        objects = [
            {
                "object_id": node["id"],
                "names": node["names"],
                "attributes": node.get("attributes", []),
            }
            for node in input_dict["nodes"]
        ]

        relationships = [
            {
                "subject_id": edge["subject_id"],
                "object_id": edge["object_id"],
                "predicate": edge["predicate"]
            }
            for edge in input_dict["edges"]
        ]

        data = {"objects": objects, "relationships": relationships}
        #objects = self.objects.get(image_id, [])
        #attributes = self.attributes.get(image_id, [])
        #relationships = self.relationships.get(image_id, [])
        image_width = 224
        image_height = 224
        
        return self.graph_constructor.construct_graph(data, image_width, image_height)


def custom_collate_fn(batch):
    idx = [item[0]  for item in batch]
    image_inputs = [item[1] for item in batch]  
    graph_data = [item[2] for item in batch] 

    pixel_values = [item['pixel_values'].squeeze(0) for item in image_inputs]
    pixel_values = torch.stack(pixel_values, dim=0)
    image_inputs = {'pixel_values': pixel_values}

    with torch.no_grad():
        batched_graph = GraphBatch.from_data_list(graph_data)

    return idx, image_inputs, batched_graph



####
####
####
####
####
####
####
####
####

######### Models for CLIP_GNN #############



class GATPoolLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GATPoolLayer, self).__init__()
        self.attention_layer = GATConv(input_dim, output_dim)
        self.proj = torch.nn.Linear(output_dim, 1)  # Projection layer to compute attention scores

    def forward(self, x, edge_index, batch):
        x = self.attention_layer(x, edge_index) 

        attention_scores = self.proj(x)  
        attention_scores = torch.sigmoid(attention_scores)  

        
        x_weighted = attention_scores * x  # Shape: [total_num_nodes, output_dim]

        pooled = global_mean_pool(x_weighted, batch)  # Shape: [batch_size, output_dim]

        return pooled, attention_scores


class GNNTextEncoderWithGATPool(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNTextEncoderWithGATPool, self).__init__()
        self.gat_conv1 = GATConv(input_dim, hidden_dim, edge_dim=hidden_dim)
        self.gat_conv2 = GATConv(hidden_dim, output_dim, edge_dim=hidden_dim)
        
        self.edge_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.node_proj = nn.Linear(input_dim, input_dim)
        
        self.pool1 = GATPoolLayer(output_dim, output_dim)
        self.pool2 = GATPoolLayer(output_dim, output_dim)


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        x = F.relu(self.gat_conv1(x, edge_index, edge_attr))
        x = F.relu(self.gat_conv2(x, edge_index, edge_attr))

        graph_embedding1, _ = self.pool1(x, edge_index, batch)
        graph_embedding2, _ = self.pool2(x, edge_index, batch)

        graph_embedding = torch.cat([graph_embedding1, graph_embedding2], dim=1)

        return graph_embedding
        

class ClipWithGNN(nn.Module):
    def __init__(self, clip_model, gnn_model):
        super(ClipWithGNN, self).__init__()
        self.clip_model = clip_model
        self.gnn_model = gnn_model

        for param in self.clip_model.parameters():
            param.requires_grad = False

        clip_image_dim = self.clip_model.vision_model.config.hidden_size
        self.image_projection = nn.Linear(512, 1024)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))  

    def forward(self, image_inputs, graph_data):
        image_embeds = self.clip_model.get_image_features(**image_inputs)
        image_embeds = self.image_projection(image_embeds)
        #print(image_embeds.shape)
        text_embeds = self.gnn_model(graph_data)
        #print(text_embeds.shape)
        
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(image_embeds, text_embeds.T)
        
        return logits
    def get_image_embeddings(self, image_inputs):
        return self.clip_model.get_image_features(**image_inputs)
    def get_graph_embeddings(self, graph_data):
        return self.gnn_model(graph_data)




####
####
####
####
####
####
####
####
####

######### Training and Loss for CLIP_GNN #############


def save_checkpoint(model, optimizer, epoch, batch_idx, save_path="CLIP_GNN.pt"):
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)

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

def train(model, dataloader, optimizer, device, accumulation_steps=8, grad_clip_value=1.0, save_path="CLIP_GNN.pt"):
    model.train()
    total_loss = 0.0
    accumulated_steps = 0

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if not batch:
            continue
        
        idx, image_inputs, graph_data = batch
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        graph_data = graph_data.to(device)

        logits = model(image_inputs, graph_data)
        batch_size = logits.shape[0]

        if logits.shape[0] != logits.shape[1]:
            print("logits not square")
            continue

        loss = contrastive_loss(logits, batch_size)

        loss = loss / accumulation_steps  
        loss.backward()

        accumulated_steps += 1
        
        if accumulated_steps % accumulation_steps == 0:


            optimizer.step()

            optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            save_checkpoint(model, optimizer, 0, batch_idx, save_path)


        del image_inputs, graph_data, logits, loss  # Free memory
        torch.cuda.empty_cache()

    total_loss /= len(dataloader)
    print(f"Training loss: {total_loss}")



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

######### Getting Main Results CLIP_GNN #############


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
    objects_data = load_json('objects.json')
    attributes_data = load_json('attributes.json')
    relationships_data = load_json('relationships.json')
    synsets_data = load_json('synsets.json')  

    image_data_dict = {item['image_id']: item for item in image_data}
    objects_data_dict = {item['image_id']: item['objects'] for item in objects_data}
    attributes_data_dict = {item['image_id']: item['attributes'] for item in attributes_data}
    relationships_data_dict = {item['image_id']: item['relationships'] for item in relationships_data}
    synsets_data_dict = {item['synset_name']: item['synset_definition'] for item in synsets_data}

    image_ids = list(image_data_dict.keys())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ids, test_ids = train_test_split(image_ids, test_size = 0.2, random_state = 42)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    #device = "cpu"
    batch_size = 32
    train_dataset = GraphClipDataset(
                    train_ids, 
                    processor, 
                    clip_model,
                    image_data_dict,
                    objects_data_dict,
                    attributes_data_dict,
                    relationships_data_dict,
                    synsets_data_dict,
                    )

    test_dataset = GraphClipDataset(
                    test_ids, 
                    processor, 
                    clip_model,
                    image_data_dict,
                    objects_data_dict,
                    attributes_data_dict,
                    relationships_data_dict,
                    synsets_data_dict,
                    )


    train_dataloader = DataLoader(train_dataset,
                                    batch_size = batch_size,
                                    shuffle = True,
                                    collate_fn = custom_collate_fn)


    test_dataloader = DataLoader(test_dataset,
                                    batch_size = batch_size,
                                    shuffle = False,
                                    collate_fn = custom_collate_fn)



    gnn_model = GNNTextEncoderWithGATPool(input_dim=512, hidden_dim=256, output_dim=512).to(device)
    clip_gnn_model = ClipWithGNN(clip_model, gnn_model).to(device)
    optimizer = torch.optim.Adam(clip_gnn_model.parameters(), lr=1e-4)
    print("training model")
    for epoch in range(10):
        train(clip_gnn_model, train_dataloader, optimizer, device)
    graph_dict = {'model': clip_gnn_model, 
            'dataset': test_dataset, 
            'dataloader': test_dataloader}
    output_prefix = 'CLIP_GNN'

    results = compute_similarities_with_statistics(
        graph_text_dict=graph_dict,
        device=device,
        output_prefix=output_prefix,
        top_k=[1, 5, 10]
    )
        


if __name__ == "__main__":
    main()
    

