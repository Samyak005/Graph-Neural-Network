## Part 1

### The GNN model can be summarized as:

1. Perform two graph convolutions (self.conv_1 and self.conv_2)
2. Pool all the node embeddings via global_sum_pool
3. Run the pooled embeddings through two fully connected layers (self.fc_1 and self.out_layer)
4. Output a class-membership probability via softmax

### Graph Convolution Layer
With graph convolutions, we can interpret each row of X as being an embedding of the information that is stored at the node corresponding to that row. Graph convolutions update the embeddings at each node based on the embeddings of their neighbors and themselves.

To implement the sum over neighbors in matrix form we utilize the adjacency matrix A. The matrix form of graph convolution is XW1 + AXW2. Here, the adjacency matrix, consisting of 1s and 0s, acts as a mask to select nodes and compute the desired sums. X * W1 is the update component from the node itself. A * X * W2 is the update component from message passing of neighbours.  

The forward() method implements the matrix form of the forward pass, with the addition of a bias term.

### Global Sum Pool 
global_sum_pool() implements a global pooling layer. Global pooling layers aggregate all of a graph’s node embeddings into a fixed-sized output. It sums all the node embeddings of a graph. This global pooling is similar to the global average pooling used in CNNs, which is used before the data is run through fully connected layers.

Global pooling can be done with any permutation invariant function, for example, sum, max, and mean.

## Part 2
### Dataset
PyTorch Geometric has been used to load the citeseer dataset. 

### Model
There are 2 classes - 
1. GCNConv: This class implements the convolutional layer of GCN.
2. CiteSeerNet: This class describes the model architecture.

### Results
After No adjacency matrix normalization and 600 epochs - Training Loss: 0.015 Val accuracy: 46.80\
After Symmetric adjacency matrix normalization and 800 epochs - Training Loss: 0.005 Val accuracy: 36.80\
After Row adjacency matrix normalization and 600 epochs - Training Loss: 0.0097 Val accuracy: 39.20\
After column adjacency matrix normalization and 600 epochs - Training Loss: 0.0064 Val accuracy: 39.0

Best validation accuracy is observed without any adjacency matrix normalization. 

Run:\
```bash GCN.sh```\
or\
```python3 GCN.py```

## Part 3
### Dataset
Similar to part 2, PyTorch Geometric has been used to load the citeseer dataset.

### Model
There are 3 classes:
1. NeighbourAggregator: Implements mean and max pooling neighbourhood aggregation function
2. SageGCN: Implements convolutional layer of GraphSAGE\
$h_{v}^{k}=\sigma[W^{(k)}.CONCAT(h_{v}^{k-1},AGGREGATE_{k}( \{h_{u}^{k-1} \forall u \in N(v)\}))]$
3. GraphSage: Describes the model architecture.

2 types of aggregation function have been defined - mean and max pooling

### Arguments
Arguments can be passed to choose number of layers and the aggregation function\
ARG1 is number of layers which can {2, 3}.\
ARG2 is the aggregation function which can {"mean", "pool_max"}.

Run:\
```bash GraphSage.sh [ARG1] [ARG2]```\
or\
```python3 GraphSage.py [ARG1] [ARG2]```

### Results
| Aggregation Function | Number of Layers | Test Accuracy |
|----------------------|------------------|---------------|
| Mean                 | 2                | 57.4          |
| Max Pool             | 2                | 56.4          |
| Mean                 | 3                | 59.80         |
| Max Pool             | 3                | 62.70         |