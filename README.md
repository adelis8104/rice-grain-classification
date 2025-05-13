# rice-grain-classification
Comparative Analysis of Machine Learning Models for Rice Grain Classification 
Dependencies:
Tensorflow: requires python 3.9 - 3.5

opencv: may need to run these commands if not working after going to 3.9
pip install opencv-python
pip install opencv-contrib-python

After installation, test it with:
python -c "import cv2; print(cv2.__version__)"

## 🛠️ Installation & Setup

To run this rice grain classification project, ensure you have Python 3.7 or higher installed. Follow the steps below to install the necessary dependencies and prepare the dataset.

---

### 📦 Required Downloads

1. **Rice Image Dataset**
   - Download from: [https://www.muratkoklu.com/datasets/](https://www.muratkoklu.com/datasets/)
   - Unzip and place the dataset in a directory such as:
     ```
     rice-grain-classification/Rice_Image_Dataset/
     ```

2. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/rice-grain-classification.git
   cd rice-grain-classification
