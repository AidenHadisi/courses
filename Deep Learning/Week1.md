#### What is Deep Learning?

- **Deep Learning** involves training **Neural Networks**, often large and complex networks.

#### Neural Networks Basics

- A **Neural Network** is a function that maps input data (xx) to output data (yy) using interconnected processing units called neurons.

---

### Example: Housing Price Prediction

1. **Dataset**:
    
    - Input (xx): Features such as size of houses.
    - Output (yy): House prices.
2. **Linear vs Non-linear Fit**:
    
    - Linear regression fits a straight line, but it might predict impossible values like negative prices.
    - Using a **ReLU function** (Rectified Linear Unit): A function that outputs max(0,ax+b)\text{max}(0, ax + b), ensuring non-negative predictions.
3. **Simplest Neural Network**:
    
    - Input (xx): House size.
    - Output (yy): Predicted price.
    - The network has one neuron implementing the ReLU function.

---

### Expanding the Neural Network

1. **Additional Features**:
    - Features like number of bedrooms, zip code, and neighborhood wealth are added.
    - These features influence intermediate factors such as:
        - Family size (e.g., size and bedrooms).
        - Walkability (e.g., zip code).
        - School quality (e.g., wealth and zip code).
2. **Neurons as Building Blocks**:
    - Each neuron (e.g., ReLU unit) processes inputs to compute intermediate outputs.
    - A **larger neural network** stacks neurons to form complex functions.
3. **Densely Connected Layers**:
    - All input features (x1,x2,...,xnx_1, x_2, ..., x_n) are connected to every neuron in the hidden layer.
    - The network learns relationships between inputs and outputs autonomously. (i.e. we don't give the model Family size, Walkability, School quality. Instead, the model figures these out by itself based on our features.)

---

### Key Components of Neural Networks

- **Input Layer**: Accepts all features (xx).
- **Hidden Layer**: Composed of neurons (hidden units), each taking all input features. (Not just some features, but all features are passed to each hidden unit)
- **Output Layer**: Produces the final prediction (yy).

---

### Training Neural Networks

- Provide examples with known inputs (xx) and outputs (yy) from the dataset.
- The network learns to map inputs to outputs by adjusting weights in neurons.

---

### Applications

- Neural networks excel in **supervised learning**, mapping inputs (xx) to outputs (yy).
- Example: Predicting housing prices from features.

---

# Neural Networks and Supervised Learning

### Overview

Neural networks have generated significant hype due to their remarkable performance in various applications. Most of the economic value created by neural networks thus far stems from **supervised learning**, **a type of machine learning where a function is learned to map input xx to output yy.**

---

### Applications of Supervised Learning

1. **Housing Price Prediction**  
    Input: Features of a house (e.g., size, location)  
    Output: Estimated price (yy)
2. **Online Advertising**
    - Most lucrative application.
    - Input: User and ad information.
    - Output: Probability of clicking on an ad.
    - Impact: Drives significant revenue for online advertising companies.
3. **Computer Vision**
    - Applications: Photo tagging, object recognition.
    - Input: Image.
    - Output: Classification (e.g., 1–1,000 categories).
4. **Speech Recognition**
    - Input: Audio clip.
    - Output: Text transcript.
5. **Machine Translation**
    - Input: English sentence.
    - Output: Translated sentence (e.g., Chinese).
6. **Autonomous Driving**
    - Input: Images and radar data.
    - Output: Positions of other cars and objects.

---

### Neural Network Architectures

1. **Standard Neural Networks**
    - General-purpose architecture, suitable for applications like real estate and online advertising.
2. **Convolutional Neural Networks (CNNs)**
    - Designed for image data.
    - Extract spatial hierarchies of features.
3. **Recurrent Neural Networks (RNNs)**
    - Ideal for sequence data (e.g., audio, text, language)
    - Captures temporal dependencies.
4. **Hybrid Architectures**
    - Used for complex applications like autonomous driving (e.g., combining CNNs for images and custom layers for radar data).

---

### Structured vs. Unstructured Data

1. **Structured Data**
    - Examples: Databases with **clear, well-defined features** (e.g., size of a house, user age).
    - Applications: Predictive analytics, advertising systems.
    - Contribute significantly to economic value.
1. **Unstructured Data**
    - Examples: Images, audio, text.
    - Features are raw (e.g., pixel values, words).
    - Neural networks have significantly improved performance in understanding unstructured data.
    - Often attract media attention due to their intuitive appeal (e.g., recognizing cats in images).

--- 

# The Rise of Deep Learning

**1. Introduction to the Rise of Deep Learning**
- **Question**: Why has deep learning taken off despite the ideas being decades old?
- **Key Insight**: Understanding the drivers behind the rise of deep learning helps identify opportunities for application in organizations.

**2. The Role of Data and Performance**
- **Data Availability**:
  - Historically, limited data restricted algorithm performance.
  - The digitization of society has massively increased available data through:
    - Digital activities (websites, mobile apps).
    - Sensors (cameras, accelerometers, IoT devices).
- **Performance and Data**:
  - Traditional algorithms like SVM and logistic regression plateau in performance with increasing data.
  - Neural networks improve performance as more data is provided, especially larger networks.

**3. Key Drivers of Deep Learning Progress**
- **Scale**:
  - **Data Scale**: A large amount of labeled data (“m” = training set size) is essential.
  - **Model Scale**: Larger neural networks with more parameters and connections drive better performance.
- **Algorithmic Innovations**:
  - Transition from sigmoid to ReLU activation functions improved gradient descent efficiency.
  - Faster algorithms have enabled quicker training and experimentation.
- **Computation**:
  - Specialized hardware like GPUs and fast networking has accelerated training.
  - Faster computation enables iterative experimentation and innovation.

**4. Challenges and Considerations**
- **Small Data Regimes**:
  - With limited data, traditional algorithms can outperform neural networks due to manual feature engineering.
  - Neural networks dominate in large data regimes.
- **Limits of Scale**:
  - Eventually, data or computational resources may become constraints.

**5. Experimental Cycle in Neural Network Development**
- **Workflow**:
  - Idea ➔ Code Implementation ➔ Experimentation ➔ Analysis ➔ Refinement.
- **Impact of Computation Speed**:
  - Faster training reduces iteration time, enhancing productivity and innovation.

**6. Future Prospects for Deep Learning**
- **Continuous Improvement**:
  - Data: Increasing digital data generation.
  - Computation: Advancements in hardware and networking.
  - Algorithms: Ongoing innovations from the research community.
- **Optimism**: These drivers ensure deep learning will continue to improve for years to come.



