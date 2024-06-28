# AI-by-Hand-exercises-Tom-Yeh
Excel exercises done following posts from Professor Tom Yeh (IG: @ProfTomYeh)

1_LSTM

https://x.com/ProfTomYeh/status/1806285518126317877
Exercise:
[LSTM_Tom Yeh.xlsx](https://github.com/user-attachments/files/16026954/LSTM_Tom.Yeh.xlsx)

2_Superposition
https://x.com/ProfTomYeh/status/1802839647057523158
Exercise:
[Superpositon_Tom Yeh.xlsx](https://github.com/user-attachments/files/16026960/Superpositon_Tom.Yeh.xlsx)

3_Vector database
https://x.com/ProfTomYeh/status/1795076707386360227
Exercise:
[Vector database_Tom Yeh.xlsx](https://github.com/user-attachments/files/16026963/Vector.database_Tom.Yeh.xlsx)

4_U-Net
U-Net is a convolutional neural network that was developed for biomedical image segmentation at the Computer Science Department of the University of Freiburg.[1] The network is based on a fully convolutional neural network[2] whose architecture was modified and extended to work with fewer training images and to yield more precise segmentation. Segmentation of a 512 × 512 image takes less than a second on a modern (2015) GPU using the U-Net architecture.[1]
The U-Net architecture has also been employed in diffusion models for iterative image denoising.[3] This technology underlies many modern image generation models, such as DALL-E, Midjourney, and Stable Diffusion.

https://x.com/ProfTomYeh/status/1786064644982948106
Exercise:

5_GAN
A generative adversarial network (GAN) is a class of machine learning frameworks and a prominent framework for approaching generative AI.[1][2] The concept was initially developed by Ian Goodfellow and his colleagues in June 2014.[3] In a GAN, two neural networks contest with each other in the form of a zero-sum game, where one agent's gain is another agent's loss.
Given a training set, this technique learns to generate new data with the same statistics as the training set. For example, a GAN trained on photographs can generate new photographs that look at least superficially authentic to human observers, having many realistic characteristics. Though originally proposed as a form of generative model for unsupervised learning, GANs have also proved useful for semi-supervised learning,[4] fully supervised learning,[5] and reinforcement learning.[6]
The core idea of a GAN is based on the "indirect" training through the discriminator, another neural network that can tell how "realistic" the input seems, which itself is also being updated dynamically.[7] This means that the generator is not trained to minimize the distance to a specific image, but rather to fool the discriminator. This enables the model to learn in an unsupervised manner.
GANs are similar to mimicry in evolutionary biology, with an evolutionary arms race between both networks.

https://x.com/ProfTomYeh/status/1786124446303989880
Exercise:

6_Backpropagation
In machine learning, backpropagation is a gradient estimation method used to train neural network models. The gradient estimate is used by the optimization algorithm to compute the network parameter updates.
It is an efficient application of the chain rule to neural networks.[1] It is also known as the reverse mode of automatic differentiation or reverse accumulation, due to Seppo Linnainmaa (1970).[2][3][4][5][6][7][8] The term "back-propagating error correction" was introduced in 1962 by Frank Rosenblatt,[9][1] but he did not know how to implement this, even though Henry J. Kelley had a continuous precursor of backpropagation[10] already in 1960 in the context of control theory.[1]
Backpropagation computes the gradient of a loss function with respect to the weights of the network for a single input–output example, and does so efficiently, computing the gradient one layer at a time, iterating backward from the last layer to avoid redundant calculations of intermediate terms in the chain rule; this can be derived through dynamic programming.[10][11][12] Gradient descent, or variants such as stochastic gradient descent,[13] are commonly used.

https://x.com/ProfTomYeh/status/1787874286813733108
Exercise:

7_RRN
A recurrent neural network (RNN) is one of the two broad types of artificial neural network, characterized by direction of the flow of information between its layers. In contrast to the uni-directional feedforward neural network, it is a bi-directional artificial neural network, meaning that it allows the output from some nodes to affect subsequent input to the same nodes. Their ability to use internal state (memory) to process arbitrary sequences of inputs[1][2][3] makes them applicable to tasks such as unsegmented, connected handwriting recognition[4] or speech recognition.[5][6] The term "recurrent neural network" is used to refer to the class of networks with an infinite impulse response, whereas "convolutional neural network" refers to the class of finite impulse response. Both classes of networks exhibit temporal dynamic behavior.[7] A finite impulse recurrent network is a directed acyclic graph that can be unrolled and replaced with a strictly feedforward neural network, while an infinite impulse recurrent network is a directed cyclic graph that cannot be unrolled.
Additional stored states and the storage under direct control by the network can be added to both infinite-impulse and finite-impulse networks. Another network or graph can also replace the storage if that incorporates time delays or has feedback loops. Such controlled states are referred to as gated states or gated memory and are part of long short-term memory networks (LSTMs) and gated recurrent units. This is also called Feedback Neural Network (FNN). Recurrent neural networks are theoretically Turing complete and can run arbitrary programs to process arbitrary sequences of inputs.[8]
https://x.com/ProfTomYeh/status/1787906449227554917
Exercise:

8_Autoencoder
An autoencoder is a type of artificial neural network used to learn efficient codings of unlabeled data (unsupervised learning).[1][2] An autoencoder learns two functions: an encoding function that transforms the input data, and a decoding function that recreates the input data from the encoded representation. The autoencoder learns an efficient representation (encoding) for a set of data, typically for dimensionality reduction.
Variants exist, aiming to force the learned representations to assume useful properties.[3] Examples are regularized autoencoders (Sparse, Denoising and Contractive), which are effective in learning representations for subsequent classification tasks,[4] and Variational autoencoders, with applications as generative models.[5] Autoencoders are applied to many problems, including facial recognition,[6] feature detection,[7] anomaly detection and acquiring the meaning of words.[8][9] Autoencoders are also generative models which can randomly generate new data that is similar to the input data (training data).[7]

https://x.com/ProfTomYeh/status/1788238035080900904
Exercise:

9_Variational Auto Encoder
In machine learning, a variational autoencoder (VAE) is an artificial neural network architecture introduced by Diederik P. Kingma and Max Welling.[1] It is part of the families of probabilistic graphical models and variational Bayesian methods.[2]
In addition to being seen as an autoencoder neural network architecture, variational autoencoders can also be studied within the mathematical formulation of variational Bayesian methods, connecting a neural encoder network to its decoder through a probabilistic latent space (for example, as a multivariate Gaussian distribution) that corresponds to the parameters of a variational distribution.
Thus, the encoder maps each point (such as an image) from a large complex dataset into a distribution within the latent space, rather than to a single point in that space. The decoder has the opposite function, which is to map from the latent space to the input space, again according to a distribution (although in practice, noise is rarely added during the decoding stage). By mapping a point to a distribution instead of a single point, the network can avoid overfitting the training data. Both networks are typically trained together with the usage of the reparameterization trick, although the variance of the noise model can be learned separately.[citation needed]
Although this type of model was initially designed for unsupervised learning,[3][4] its effectiveness has been proven for semi-supervised learning[5][6] and supervised learning.[7]

https://x.com/ProfTomYeh/status/1788299547355021719
Exercise:

10_Transformer
A transformer is a deep learning architecture developed by Google and based on the multi-head attention mechanism, proposed in a 2017 paper "Attention Is All You Need".[1] Text is converted to numerical representations called tokens, and each token is converted into a vector via looking up from a word embedding table.[1] At each layer, each token is then contextualized within the scope of the context window with other (unmasked) tokens via a parallel multi-head attention mechanism allowing the signal for key tokens to be amplified and less important tokens to be diminished. The transformer paper, published in 2017, is based on the softmax-based attention mechanism proposed by Bahdanau et. al. in 2014 for machine translation,[2][3] and the Fast Weight Controller, similar to a transformer, proposed in 1992.[4][5][6]
Transformers have the advantage of having no recurrent units, and therefore require less training time than earlier recurrent neural architectures such as long short-term memory (LSTM).[7] Later variations have been widely adopted for training large language models (LLM) on large (language) datasets, such as the Wikipedia corpus and Common Crawl.[8]
This architecture is now used not only in natural language processing and computer vision,[citation needed] but also in audio,[9] multi-modal processing and robotics.[10] It has also led to the development of pre-trained systems, such as generative pre-trained transformers (GPTs)[11] and BERT[12] (Bidirectional Encoder Representations from Transformers).

https://x.com/ProfTomYeh/status/1794702829103309291
Exercise:

11_SORA
Sora is a diffusion model that starts with a video that resembles static noise. Over many steps, the output gradually transforms by removing the noise. By providing the model with the foresight of multiple frames concurrently, OpenAI has resolved the complex issue of maintaining subject consistency, even when it momentarily disappears from view.

https://x.com/ProfTomYeh/status/1795449683285848509
Exercise:

12_RLHF
In machine learning, reinforcement learning from human feedback (RLHF) is a technique to align an intelligent agent to human preferences. In classical reinforcement learning, the goal of such an agent is to learn a function that guides its behavior called a policy. This function learns to maximize the reward it receives from a separate reward function based on its task performance.[1] However, it is difficult to define explicitly a reward function that approximates human preferences. Therefore, RLHF seeks to train a "reward model" directly from human feedback.[2] The reward model is first trained in a supervised fashion—independently from the policy being optimized—to predict if a response to a given prompt is good (high reward) or bad (low reward) based on ranking data collected from human annotators. This model is then used as a reward function to improve an agent's policy through an optimization algorithm like proximal policy optimization.[3]
RLHF has applications in various domains in machine learning, including natural language processing tasks such as text summarization and conversational agents, computer vision tasks like text-to-image models, and the development of video game bots. While RLHF is an effective method of training models to act better in accordance with human preferences, it also faces challenges due to the way the human preference data is collected. Though RLHF does not require massive amounts of data to improve performance, sourcing high-quality preference data is still an expensive process. Furthermore, if the data is not carefully collected from a representative sample, the resulting model may exhibit unwanted biases.

https://x.com/ProfTomYeh/status/1795803665845899511
Exercise:

13_Self Attention
The machine learning-based attention method simulates how human attention works by assigning varying levels of importance to different words in a sentence. It assigns importance to each word by calculating "soft" weights for the word's numerical representation, known as its embedding, within a specific section of the sentence called the context window to determine its importance

Self-attention, also known as scaled dot-product attention, is a fundamental concept in the field of NLP and deep learning. It plays a pivotal role in tasks such as machine translation, text summarization, and sentiment analysis. Self-attention enables models to weigh the importance of different parts of an input sequence when making predictions or capturing dependencies between words

https://x.com/ProfTomYeh/status/1797249951325434315
Exercise:
