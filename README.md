# **Coding Assignment 6: Latent-Variable Generative Models**

</br>

## **Overview**

This homework consists of five short notebooks, tracing the history of latent-variable generative modeling strategies. These all go beyond the autoregressive approach that is so successful in language modeling that we learned in the first half of the course.

Whereas the autoregressive approach factors the target distribution into a sequence of tokens (each one modeled as a conditional probability distribution that depends on the previous tokens) the latent-variable approach models the distribution as a process or a function f that transforms a boring distribution z (e.g., a random vector drawn from a normal distribution) into an interesting distribution x = f(z) that imitates the target distribution.

In the first four notebooks you will use a simple "swiss roll" distribution as the target. In all these the goal will be to learn some process f, so that x = f(z) looks like a "swiss roll" when z is drawn from a normal distribution.

You will see that in some of the methods, the f to learn is directly a neural network, whereas in other cases, f is a process that incorporates a neural network that plays a particular role within that process. A couple of the methods also train a supplementary neural network to help. All of these different schemes are designed to put the neural network in a situation that makes it possible to optimize it easily towards the goal of making f(z) imitate the target distribution, if all we have as training input is a big set of samples.

In the final notebook, you will apply diffusion modeling to make an MNIST image generative model. (If it interests you, as a purely optional exercise with no credit, at the end of that last notebook you can implement any of the other methods (GAN, RealNVP, or Flow Matching) to implement and visualize an MNIST generator.)

---

## **Notebooks**

### **6.1 - Generative Adversarial Networks (GANs)**

Implement the adversarial training objective and explore the learned latent space.

**Topics:**
- Minimax game formulation
- Discriminator and Generator loss functions
- Training stability and mode collapse
- Latent space interpolation

**Point Breakdown (15 points):**

| Component | Coding | Questions/Analysis |
|-----------|--------|--------------------|
| **Discriminator Loss** | 3 pts | — |
| **Generator Loss** | 2 pts | — |
| **Latent Interpolation** | 3 pts | — |
| **Analysis Questions** | — | 7 pts |

---

### **6.2 - Diffusion Models**

Implement both the forward (noising) and reverse (denoising) processes of a DDPM-style diffusion model.

**Topics:**
- Forward diffusion process and noise scheduling
- Reverse denoising step derivation
- Noise prediction training objective
- Computational cost analysis

**Point Breakdown (20 points):**

| Component | Coding | Questions/Analysis |
|-----------|--------|--------------------|
| **Forward Diffusion** | 4 pts | — |
| **Reverse Step** | 4 pts | — |
| **Training Loss** | 4 pts | — |
| **Analysis Questions** | — | 8 pts |

---

### **6.3 - Flow Matching**

Implement the flow matching training objective and a higher-order ODE solver for sampling.

**Topics:**
- Linear interpolation and velocity fields
- Flow matching training objective
- Euler vs. Runge-Kutta integration
- Comparison with diffusion models

**Point Breakdown (15 points):**

| Component | Coding | Questions/Analysis |
|-----------|--------|--------------------|
| **Training Objective** | 4 pts | — |
| **RK4 Solver** | 4 pts | — |
| **Analysis Questions** | — | 7 pts |

---

### **6.4 - RealNVP (Normalizing Flows)**

Implement the affine coupling layer at the heart of RealNVP and understand exact likelihood computation.

**Topics:**
- Affine coupling transformations
- Change of variables formula
- Tractable Jacobian determinants
- Exact vs. approximate likelihood

**Point Breakdown (10 points):**

| Component | Coding | Questions/Analysis |
|-----------|--------|--------------------|
| **Coupling Layer Forward** | 3 pts | — |
| **Log-Likelihood Loss** | 3 pts | — |
| **Analysis Questions** | — | 4 pts |

---

### **6.5 - MNIST Diffusion Generation**

Apply diffusion modeling to image data, implementing the training loop and improving sample quality.

**Topics:**
- Convolutional architectures for diffusion
- Training loop implementation
- Hyperparameter tuning for sample quality
- Architectural adaptations for image data

**Point Breakdown (10 points):**

| Component | Coding | Questions/Analysis |
|-----------|--------|--------------------|
| **Training Loop** | 4 pts | — |
| **Sample Improvement** | 3 pts | — |
| **Analysis Questions** | — | 3 pts |

---

## **Total Points: 70**

| Notebook | Coding | Analysis |
|----------|--------|----------|
| 6.1 GAN | 8 pts | 7 pts |
| 6.2 Diffusion | 12 pts | 8 pts |
| 6.3 Flow Matching | 8 pts | 7 pts |
| 6.4 RealNVP | 6 pts | 4 pts |
| 6.5 MNIST | 7 pts | 3 pts |
| **Total** | **41 pts** | **29 pts** |

---

## **Data Setup**

### Swiss Roll (Notebooks 6.1–6.4):
Generated synthetically in each notebook — no external data needed.

### MNIST (Notebook 6.5):
You can find the required MNIST data here: [Link to Drive](https://drive.google.com/drive/folders/1N4zu1sI7c1L1ySp04qOyasN5RdMYXiKZ?usp=sharing).

### **Installation:**
```bash
pip install torch numpy matplotlib tqdm torchvision
```

---

## **Submission Instructions**

Submit **all five completed notebooks** by the deadline.

### **Deliverables:**
- `FirstName_6.1_GAN.ipynb`
- `FirstName_6.2_Diffusion.ipynb`
- `FirstName_6.3_FlowMatching.ipynb`
- `FirstName_6.4_RealNVP.ipynb`
- `FirstName_6.5_MNIST.ipynb`

### **Submission:**

**Zipped:**
Create `FirstName_CA6.zip` containing all five notebooks

### **Requirements:**
- All TODO sections must be filled in with working code
- All questions answered in markdown cells (marked "YOUR ANSWER:")
- Generated outputs visible (loss plots, sample visualizations, comparisons)
- Code runs without errors from top to bottom

**Submission:** Submit the file(s) as a direct **reply** to the *coding assignment module* on Canvas.

---

## **Learning Objectives**

By completing this assignment, you will:
1. Implement four distinct generative modeling paradigms (GAN, Diffusion, Flow Matching, Normalizing Flows)
2. Understand the trade-offs between adversarial, score-based, flow-based, and likelihood-based training
3. Analyze computational costs of generation across different methods
4. Apply diffusion models to real image data (MNIST)
5. Compare latent-variable approaches to the autoregressive methods from CA5

---

## **Resources**

**Papers Referenced:**
- GANs: Goodfellow et al. (2014) — Generative Adversarial Networks
- Diffusion: Ho et al. (2020) — Denoising Diffusion Probabilistic Models
- Flow Matching: Lipman et al. (2023) — Flow Matching for Generative Modeling
- RealNVP: Dinh et al. (2017) — Density Estimation Using Real-valued Non-Volume Preserving Transformations

**Additional Reading:**
- [Lilian Weng's Diffusion Models Blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [What are Diffusion Models? (Tutorial)](https://arxiv.org/abs/2208.11970)
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875) by Arjovsky et al. (2017)

---

## **Academic Integrity**

- **Collaboration:** Follow course syllabus policy on collaboration
- **Citations:** Credit all external resources, discussions, and AI assistance used
- **Code reuse:** Cite any code adapted from online sources
- **Individual work:** Submit your own implementations — copying code is prohibited

---

**Good luck exploring generative models!**
