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
import torch.multiprocessing as mp


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

######### Dataset for CLIP_BB_GNN #############


def create_graph(node_features, edge_index, edge_features):
    if node_features.size(0) <= 1 or edge_index.size(1) == 0:
        #print("Invalid graph detected. Returning empty graph.")
        return Data(
            x=torch.zeros((0, node_features.size(1)), dtype=node_features.dtype),  # No nodes
            edge_index=torch.zeros((2, 0), dtype=edge_index.dtype),  # No edges
            edge_attr=torch.zeros((0, edge_features.size(1)), dtype=edge_features.dtype)  # No edge features
        )
    graph_data = Data(
        x=node_features,  # Node embeddings
        edge_index=edge_index,  # Edge connections
        edge_attr=edge_features  # Edge embeddings
    )
    return graph_data

class SceneGraphConstructor:
    def __init__(self, processor, clip_model):
        self.processor = processor
        self.clip_model = clip_model

    def construct_graph(self, data, image_width, image_height):
        objects = data['objects']
        relationships = data['relationships']

        node_features, object_id_to_idx = self._construct_node_features(objects, image_width, image_height)

        edge_index, edge_features = self._construct_edge_features(relationships, object_id_to_idx, objects, image_width, image_height)


        edge_index, edge_features = self._add_weak_edges(objects, edge_index, edge_features, object_id_to_idx, image_width, image_height)

        graph_data = create_graph(node_features, edge_index, edge_features)
        
        return graph_data

    def _construct_node_features(self, objects, image_width, image_height):
        node_features = []
        object_id_to_idx = {}
        descriptions = []

        for i, obj in enumerate(objects):
            object_id = obj['object_id']
            object_id_to_idx[object_id] = i

            bbox = [obj['x'], obj['y'], obj['w'], obj['h']]
            normalized_bbox = torch.tensor(
            [
                bbox[0] / image_width, 
                bbox[1] / image_height,
                bbox[2] / image_width, 
                bbox[3] / image_height
            ], dtype=torch.float, device=self.clip_model.device)
            node_features.append(normalized_bbox)
            obj_names = ', '.join(obj['names'])
            attributes = ', '.join(obj.get('attributes', [])) or "None"
            description = f"Object: {obj_names}, Attributes: {attributes}"
            descriptions.append(description)

        if descriptions:
            text_embeddings = self._encode_texts(descriptions).to(self.clip_model.device)
            node_features = [torch.cat([te, nf], dim=-1) for te, nf in zip(text_embeddings, node_features)]
            node_features = torch.stack(node_features)
        else:
            node_features = torch.zeros((1, 516), dtype=torch.float, device=self.clip_model.device)

        return node_features, object_id_to_idx

    def _construct_edge_features(self, relationships, object_id_to_idx, objects, image_width, image_height):
        edge_index = []
        edge_features = []
        predicates = []
        spatial_features_list = []

        for rel in relationships:
            subj_id = rel['subject_id']
            obj_id = rel['object_id']
            if subj_id not in object_id_to_idx or obj_id not in object_id_to_idx:
                continue 

            predicate = rel['predicate']
            subj_idx = object_id_to_idx[subj_id]
            obj_idx = object_id_to_idx[obj_id]

            edge_index.append([subj_idx, obj_idx])

            subj_bbox = self._get_bbox(objects, subj_id)
            obj_bbox = self._get_bbox(objects, obj_id)
            spatial_features = self._compute_spatial_features(subj_bbox, obj_bbox, image_width, image_height)
            spatial_features = spatial_features.to(self.clip_model.device)  # Move to device
            spatial_features_list.append(spatial_features)

            predicates.append(predicate)

        if predicates:
            predicate_embeddings = self._encode_texts(predicates).to(self.clip_model.device)
            edge_features = [torch.cat([pe, sf], dim=-1) for pe, sf in zip(predicate_embeddings, spatial_features_list)]
            edge_features = torch.stack(edge_features).to(self.clip_model.device)
        else:
            edge_index = torch.zeros((2, 1), dtype=torch.long).to(self.clip_model.device)   # Shape [2, 1]
            edge_features = torch.zeros((1, 515), dtype=torch.float).to(self.clip_model.device)  

        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.clone().detach()
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long)

        return edge_index.to(torch.long).t().contiguous().to(self.clip_model.device), edge_features.to(self.clip_model.device)


    def _add_weak_edges(self, objects, edge_index, edge_features, object_id_to_idx, image_width, image_height):
        if len(objects) < 2:
            return edge_index, edge_features
        if edge_index.shape[0] == 1 and edge_index.shape[1] == 2:
                edge_index = edge_index.t().contiguous()

        centers = []
        for obj in objects:
            x_center = (obj['x'] + obj['w'] / 2) / image_width
            y_center = (obj['y'] + obj['h'] / 2) / image_height
            centers.append([x_center, y_center])

        centers = np.array(centers)

        n_neighbors = min(5, len(centers))
        if n_neighbors < 2:
            return edge_index, edge_features

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        nbrs.fit(centers)
        distances, indices = nbrs.kneighbors(centers)
        edge_index = edge_index.to(self.clip_model.device)
        edge_features = edge_features.to(self.clip_model.device)

        new_edges = []
        new_edge_features = []
        feature_dim = edge_features.size(1) if edge_features.numel() > 0 else 1

        for idx, (neighbor_indices, neighbor_distances) in enumerate(zip(indices, distances)):
            for neighbor_idx, distance in zip(neighbor_indices[1:], neighbor_distances[1:]):  
                subj_idx = idx
                obj_idx = neighbor_idx
                new_edges.append([subj_idx, obj_idx])

                proximity_feature = torch.tensor([distance], dtype=torch.float)
                proximity_feature = F.pad(proximity_feature, (0, feature_dim - 1), "constant", 0)
                new_edge_features.append(proximity_feature)

        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous().to(self.clip_model.device) 
            new_edge_features = torch.stack(new_edge_features).to(self.clip_model.device) 
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"new_edge_index shape: {new_edge_index.shape}")

            if edge_index.numel() > 0:
                edge_index = torch.cat([edge_index, new_edge_index], dim=1)
            else:
                edge_index = new_edge_index

            if edge_features.numel() > 0:
                edge_features = torch.cat([edge_features, new_edge_features], dim=0)
            else:
                edge_features = new_edge_features

        return edge_index.to(self.clip_model.device), edge_features.to(self.clip_model.device)
    def _encode_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.clip_model.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**inputs)
        return text_embeddings

    def _get_bbox(self, objects, object_id):
        for obj in objects:
            if obj['object_id'] == object_id:
                return [obj['x'], obj['y'], obj['w'], obj['h']]
        return [0, 0, 0, 0]

    def _compute_spatial_features(self, bbox1, bbox2, image_width, image_height):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        delta_x = torch.tensor((x2 - x1) / image_width, dtype=torch.float)
        delta_y = torch.tensor((y2 - y1) / image_height, dtype=torch.float)

        distance = torch.sqrt(delta_x**2 + delta_y**2)

        return torch.stack([delta_x, delta_y, distance], dim=0)


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
        objects = data['objects']
        relationships = data['relationships']

        image_path = f"{self.image_dir}/{image_id}.jpg"
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            raise ValueError(f"Image not found: {image_path}")

        image_tensor = self.image_transform(image)
        image_inputs = self.processor(images=image, return_tensors="pt")

        image_width, image_height = image.size

        graph_data = self.graph_constructor.construct_graph(data, image_width, image_height)

        return image_id, image_inputs, graph_data

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

    def create_graph_from_dict(self, input_dict, image_width, image_height):
        objects = [
            {
                "object_id": node["id"],
                "names": node["names"],
                "attributes": node.get("attributes", []),
                "x": node["bbox"][0],
                "y": node["bbox"][1],
                "w": node["bbox"][2],
                "h": node["bbox"][3]
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
        return self.graph_constructor.construct_graph(data, image_width, image_height)


# def collate_fn(batch):
#     # try:
#     #     idx = [item[0]  for item in batch]
#     #     image_inputs = [item[1] for item in batch]  
#     #     graph_data = [item[2] for item in batch] 

#     #     pixel_values = [item['pixel_values'].squeeze(0) for item in image_inputs]
#     #     pixel_values = torch.stack(pixel_values, dim=0)
#     #     image_inputs = {'pixel_values': pixel_values}

#     #     with torch.no_grad():
#     #         batched_graph = GraphBatch.from_data_list(graph_data)

#     #     return idx, image_inputs, batched_graph
#     # except Exception as e: 
#     #     print(f"Collate Error:{e}")
#     #     return None
#     idx = [item[0]  for item in batch]
#     image_inputs = [item[1] for item in batch]  
#     graph_data = [item[2] for item in batch] 

#     pixel_values = [item['pixel_values'].squeeze(0) for item in image_inputs]
#     pixel_values = torch.stack(pixel_values, dim=0)
#     image_inputs = {'pixel_values': pixel_values}

#     with torch.no_grad():
#         batched_graph = GraphBatch.from_data_list(graph_data)

#     return idx, image_inputs, batched_graph
def collate_fn(batch):
    try:
        idx = [item[0] for item in batch]
        image_inputs = [item[1] for item in batch]
        graph_data = [item[2] for item in batch]  # List of graph data objects

        # Move pixel values to the appropriate device
        pixel_values = [item['pixel_values'].squeeze(0) for item in image_inputs]
        pixel_values = torch.stack(pixel_values, dim=0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        image_inputs = {'pixel_values': pixel_values}

        # Ensure all graph tensors are on the same device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for data in graph_data:
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr.to(device)
            if data.batch is not None:
                data.batch = data.batch.to(device)

        # Batch the graph data
        batched_graph = GraphBatch.from_data_list(graph_data)

        return idx, image_inputs, batched_graph
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        return None

####
####
####
####
####
####
####
####
####

######### Models for CLIP_BB_GNN #############


class GATPoolLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GATPoolLayer, self).__init__()
        self.attention_layer = GATConv(input_dim, output_dim)
        self.proj = nn.Linear(output_dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.attention_layer(x, edge_index)

        attention_scores = self.proj(x)
        attention_scores = torch.sigmoid(attention_scores)

        x_weighted = attention_scores * x

        pooled = global_mean_pool(x_weighted, batch)

        return pooled, attention_scores


class GraphEmbeddingModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):
        super(GraphEmbeddingModel, self).__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.gat1 = GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        self.gat2 = GATConv(hidden_dim, output_dim, edge_dim=hidden_dim)
        self.pool1 = GATPoolLayer(output_dim, output_dim)
        self.pool2 = GATPoolLayer(output_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = F.relu(self.gat2(x, edge_index, edge_attr))

        graph_embedding1, _ = self.pool1(x, edge_index, batch)
        graph_embedding2, _ = self.pool2(x, edge_index, batch)

        graph_embedding = torch.cat([graph_embedding1, graph_embedding2], dim=1)

        return graph_embedding


class ImageEmbeddingModel(nn.Module):
    def __init__(self, input_dim, output_dim, clipmodel):
        super(ImageEmbeddingModel, self).__init__()
        self.clip_model = clipmodel
        self.linear = nn.Linear(input_dim, output_dim)
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.clip_model.get_image_features(**x)
        return F.normalize(self.linear(x), p=2, dim=-1)

class GraphImageAlignmentModel(nn.Module):
    def __init__(self, graph_model, image_model, embedding_dim):
        super(GraphImageAlignmentModel, self).__init__()
        self.graph_model = graph_model
        self.image_model = image_model
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, image_inputs, graph_data):
        image_embeddings = self.image_model(image_inputs)
        graph_embeddings = self.graph_model(graph_data)

        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        graph_embeddings = F.normalize(graph_embeddings, p=2, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(image_embeddings, graph_embeddings.T)
        return logits
    def get_image_embeddings(self, image_inputs):
        embeddings = self.image_model(image_inputs)
        return F.normalize(embeddings, p=2, dim=-1)
    def get_graph_embeddings(self, graph_data):
        embeddings = self.graph_model(graph_data)
        return F.normalize(embeddings, p=2, dim=-1)



####
####
####
####
####
####
####
####
####

######### Training for CLIP_BB_GNN #############


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

        loss_image_to_graph = F.cross_entropy(logits, labels)
        loss_graph_to_image = F.cross_entropy(logits.T, labels)

        return (loss_image_to_graph + loss_graph_to_image) / 2

    except Exception as e:
        print(f"Exception in contrastive_loss: {e}")

def load_json(filename):
    with open(os.path.join(VG_DIR, filename), 'r') as f:
        return json.load(f)



def save_checkpoint(model, optimizer, epoch, batch_idx):
    checkpoint_filename =  f"CLIP_BB_GNN.pt"
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_filename)

def train(model, dataloader, optimizer, device, accumulation_steps=4):
    model.train()
    total_loss = 0.0
    batch_loss = 0.0
    optimizer.zero_grad()
    scaler = GradScaler()

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch:
            try:
                image_ids, image_inputs, graph_data = batch
                image_inputs = {k: v.to(device, non_blocking=True) for k, v in image_inputs.items()}
                graph_data = graph_data.to(device)

                with autocast():
                    logits = model(image_inputs, graph_data)
                    loss = contrastive_loss(logits, logits.size(0)) / accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
                batch_loss += loss.item() * accumulation_steps
                del image_inputs, graph_data, logits, loss  # Free memory
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Exception at batch {batch_idx}: {e}")
                traceback.print_exc()
                continue
            if batch_idx%100 == 0:
                save_checkpoint(model, optimizer, 0, 0)
                print("100 batch loss", batch_loss)
                batch_loss = 0.0
        else:
            continue

    return total_loss / len(dataloader)

####
####
####
####
####
####
####
####
####

######### Main Results for CLIP_BB_GNN #############
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    graph_model = GraphEmbeddingModel(node_dim=516, edge_dim=515, hidden_dim=512, output_dim=512).to(device)
    image_model = ImageEmbeddingModel(input_dim=512, output_dim=1024, clipmodel = clip_model).to(device)
    alignment_model = GraphImageAlignmentModel(graph_model, image_model, embedding_dim=512).to(device)
    image_ids = list(image_data_dict.keys())
    train_ids, test_ids = train_test_split(image_ids, test_size = 0.2, random_state = 42)
    dataset = SceneGraphDataset(
        train_ids, 
        scene_graphs,
        image_dir=image_dir,
        processor=clip_processor,
        clip_model=clip_model
    )
    dataloader = DataLoader(dataset, 
                        batch_size=32, 
                        shuffle=True, 
                        collate_fn=collate_fn,
                         num_workers=0)
    test_dataset = SceneGraphDataset(
    test_ids, 
        scene_graphs,
        image_dir=image_dir,
        processor=clip_processor,
        clip_model=clip_model
    )
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=16, 
                                shuffle=False, 
                                collate_fn=collate_fn, 
                                num_workers=0)
    optimizer = torch.optim.Adam(alignment_model.parameters(), lr=1e-4)
    for epoch in range(10):
        loss = train(alignment_model, dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    
    graph_text_dict = {'model': alignment_model, 
            'dataset': test_dataset, 
            'dataloader': test_dataloader}
    output_prefix = 'CLIP_BB_GNN'
    results = compute_similarities_with_statistics(
        graph_text_dict=graph_text_dict,
        device=device,
        output_prefix=output_prefix,
        top_k=[1, 5, 10]
    )

if __name__ == "__main__":
    main()
    