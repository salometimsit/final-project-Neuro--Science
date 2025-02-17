# 📝 Hebbian Neural Network for Letter Classification

<div align="center">

# 🧠 Letter Classification Neural Network

<h2>
A Sophisticated Implementation of Hebbian Learning with Advanced Visualization
</h2>

<p align="center">
    <b>🎯 Advanced Pattern Recognition | 🔍 Robust Classification | 📊 Comprehensive Analysis & Visualization</b>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.x-blue.svg" alt="Python 3.x">
    <img src="https://img.shields.io/badge/NumPy-Latest-green.svg" alt="NumPy">
    <img src="https://img.shields.io/badge/Matplotlib-Latest-orange.svg" alt="Matplotlib">
</p>

---

</div>

## 🌟 Overview

This project showcases an advanced implementation of a Hebbian Learning Network designed to classify English alphabet letters. The network demonstrates impressive pattern recognition capabilities, able to handle both clean and noisy inputs while maintaining robust performance. The project includes comprehensive visualization tools for analyzing network performance and behavior.

## 🎯 Key Features

### Classification Modes
- ✨ **Group Classification**: Sorts letters into three categories (A-I, J-R, S-Z)
- 🎯 **Individual Letter Recognition**: Identifies specific letters (A through Z)

### Input Flexibility
- 📋 Standard letter format
- 📝 Bold letters
- ⭕ Bold with circular styling
- 🔵 Extra bold with circular styling

### Advanced Analysis & Visualization
- 📊 Real-time training progress monitoring
- 📈 Performance metrics visualization
- 🎨 Network architecture visualization
- 📉 Noise impact analysis
- 🔍 Statistical measure tracking

### Comprehensive Testing
- 🧪 Noise tolerance testing (5%, 10%, 15%, 20%)
- 📊 Accuracy analysis across different letter styles
- 🎨 Color-coded visual feedback
- 📈 Variance and standard deviation analysis

## 🏗️ Project Structure

### 📁 Core Components

```
└── 📂 Project Root
    ├── 📜 HebbianNetwork.py     # Core neural network implementation
    ├── 📜 VectorsFactory.py     # Data generation and preprocessing
    ├── 📜 Main.py              # Execution and interface
    ├── 📜 Visualization.py     # Visualization and analysis tools
    └── 📜 README.md            # Documentation
```

### 🔧 Technical Architecture

#### HebbianNetwork.py
- 🧠 Neural network core implementation
- ⚙️ Weight matrix management
- 🎯 Training and prediction logic
- 📊 Sigmoid activation function

#### VectorsFactory.py
- 🎨 Letter vector generation
- 📚 Dataset creation
- 🔄 Noise simulation
- 📋 Output matrix management

#### Main.py
- 🎮 User interface
- 📊 Results visualization
- 📈 Statistical analysis
- 🔍 Testing procedures

#### Visualization.py
- 📊 Training progress plots
- 🎨 Network architecture visualization
- 📈 Performance analysis graphs
- 📉 Noise impact visualization

## 📊 Visualization Components

### 1. Network Architecture Visualization
- Visual representation of network topology
- Input layer (64 neurons) and output layer (3 neurons) visualization
- Connection weight visualization
- Interactive node exploration

### 2. Training Progress Visualization
- Real-time error rate tracking
- Accuracy progression graphs
- Learning curve analysis
- Convergence monitoring

### 3. Performance Analysis
- Accuracy comparison across letter groups
- Noise impact visualization
- Statistical measures plotting
- Cross-validation results

### 4. Statistical Analysis
- Standard deviation tracking
- Variance analysis
- Error distribution visualization
- Performance metrics comparison

## 💻 Technical Specifications

### Network Architecture
- **Input Layer**: 64 neurons (8x8 grid)
- **Output Layer**: 
  - 3 neurons (group classification)
  - 26 neurons (individual letters)
- **Parameters**:
  - Learning Rate: 1.0
  - Training Epochs: 5000
  - Regularization: 0.00001

### 🎨 Letter Representation
Letters are encoded as 8x8 binary matrices where:
```
1 = Letter segment
0 = Background
```

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- NumPy library
- Matplotlib library

### Installation Steps

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install dependencies:
```bash
pip install numpy matplotlib
```

### 🎮 Usage

1. Run the main script:
```bash
python Main.py
```

2. Choose your classification mode:
```
Enter a number for active the Hebbian Network:
1. For categories for 3 groups (A-I),(J-R),(S-Z)
2. For categories each letter for itself
3. Exit
```

3. Generate visualizations:
```bash
python Visualization.py
```

## 📊 Performance Analysis

### Accuracy Metrics
- Training Set: 100%
- Bold Letters: 57%
- Bold & Circle: 61%
- Extra Bold: 51%

### Noise Impact
- 5% Noise: 94.08% accuracy
- 10% Noise: 83.86% accuracy
- 15% Noise: 74.64% accuracy
- 20% Noise: 67.61% accuracy

### Statistical Measures
- Variance ranges from 0.22 to 0.79
- Standard deviation ranges from 4.69 to 8.88

## 🔍 Implementation Details

### Training Process
1. 📝 Weight matrix initialization
2. 🔄 Iterative training through epochs
3. 🧮 Hebbian learning rule application
4. ⚖️ Weight updates with regularization
5. 📈 Convergence monitoring

### Visualization Process
1. 📊 Real-time data collection
2. 📈 Dynamic plot generation
3. 🎨 Interactive visualization updates
4. 📉 Performance metric tracking

## 🛠️ Error Handling

The system includes robust error handling for:
- ✅ Input validation
- 🔍 Parameter verification
- 💪 Noise resilience
- 📉 Graceful performance degradation

## 🌟 Examples

### Sample Output
```
The accuracy with Group of the train letters: 100.00%
Results for this group:
    The letter A Suppose to be in: A-I, and the answer is: A-I
    The letter B Suppose to be in: A-I, and the answer is: A-I
    ...
```

## 👥 Contributors

This project was developed by:
- **Itay Segev**
- **Salome Timsit**

---

<div align="center">

📧 *For questions or support, please open an issue in the repository*

</div>