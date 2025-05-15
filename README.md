## Rice Grain Classification

### Installation & Setup

To run this rice grain classification project, ensure you have Python 3.7 or higher installed. Follow the steps below to install the necessary dependencies and prepare the dataset.

---

#### 1. Clone the Repository

```bash
git clone https://github.com/Jaypee2109/rice-grain-classification.git
cd rice-grain-classification
```

#### 2. Required Downloads

- **Rice Image Dataset**

  1. Download from: https://www.muratkoklu.com/datasets/
  2. Unzip and place the dataset in a directory such as:

     ```
     rice-grain-classification/Image_Dataset/
     ```

#### 3. Python Dependencies

```bash
pip install -r requirements.txt
```

- TensorFlow: requires Python 3.5 – 3.9
- OpenCV: if you encounter issues, install:

  ```bash
  pip install opencv-python
  pip install opencv-contrib-python
  ```

#### 4. Run the main file

```bash
python src/project.py
```

#### 5. Tuning Parameters

Additional hyper-parameters can be tested by extending the param_grid block in `knn.py`, `svm.py`, or `rf.py` as needed. The optimal parameters are determined automatically via grid search and attached to the returned model’s best_params.

---

## Using Docker

We also provide a Docker setup to simplify environment configuration and ensure reproducibility.

### Prerequisites

- Docker and Docker Compose installed on your machine.
- CPU-only usage by default (GPU support can be enabled).

### Build and Run

1. **Build the Docker image**

   ```bash
   docker-compose build
   ```

2. **Run the container**

   ```bash
   docker-compose up
   ```

   This will:

   - Mount `Image_Dataset` and `Rice_Subset_20` into `/app/Image_Dataset` and `/app/Rice_Subset_20` inside the container.
   - Mount `Results` into `/app/Results` to persist outputs.
   - Install all Python dependencies from `requirements.txt`.

3. **Stop the container**

   ```bash
   docker-compose down
   ```

### Customization

- **GPU support**: To enable GPU acceleration, update the `Dockerfile` base image to an NVIDIA CUDA variant and add `runtime: nvidia` under the service in `docker-compose.yml`.
- **Environment variables**: Set any needed variables in the `environment` section of `docker-compose.yml`.

---
